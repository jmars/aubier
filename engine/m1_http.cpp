// engine/m1_http.cpp — accept loop, per-connection pthreads, request parsing (see m1_http.h).

#include "m1_http.h"

#include "m1_json.h"

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <new>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

namespace m1http {

// ---------------------------------------------------------------------------
// process-wide state
// ---------------------------------------------------------------------------
static std::atomic<bool> g_stop{false};
static std::atomic<int>  g_inflight{0};
static std::mutex        g_mu;
static std::condition_variable g_cv;
static int               g_sigpipe[2] = {-1, -1};

bool stopping() { return g_stop.load(std::memory_order_relaxed); }

std::string error_body(const char * type, const char * message, const char * param,
                       const char * code, const char * extra) {
    std::string b = "{\"error\":{\"message\":";
    m1json::append_string(b, message ? message : "");
    b += ",\"type\":";
    if (type) m1json::append_string(b, type); else b += "null";
    b += ",\"param\":";
    if (param) m1json::append_string(b, param); else b += "null";
    b += ",\"code\":";
    if (code) m1json::append_string(b, code); else b += "null";
    if (extra && *extra) {
        b += ",";
        b += extra;
    }
    b += "}}";
    return b;
}

// ---------------------------------------------------------------------------
// low level I/O
// ---------------------------------------------------------------------------
static bool send_all(int fd, const char * p, size_t n) {
    size_t done = 0;
    while (done < n) {
        const ssize_t w = ::write(fd, p + done, n - done);
        if (w > 0) { done += (size_t) w; continue; }
        if (w < 0 && errno == EINTR) continue;
        return false;  // EPIPE / ECONNRESET / anything else: the peer is gone
    }
    return true;
}

static const char * reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 413: return "Content Too Large";
        case 422: return "Unprocessable Entity";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "Unknown";
    }
}

bool Conn::head(int status, const char * content_type, const char * extra_headers,
                long long content_length) {
    if (dead_) return false;
    std::string h;
    char line[96];
    std::snprintf(line, sizeof(line), "HTTP/1.1 %d %s\r\n", status, reason(status));
    h += line;
    h += "Server: aubier-m1-server/0.1\r\n";
    if (content_type && *content_type) {
        h += "Content-Type: ";
        h += content_type;
        h += "\r\n";
    }
    if (content_length >= 0) {
        std::snprintf(line, sizeof(line), "Content-Length: %lld\r\n", content_length);
        h += line;
    }
    h += "Connection: close\r\n";
    if (extra_headers) h += extra_headers;
    h += "\r\n";
    return put(h);
}

bool Conn::peer_gone() const {
    if (fd_ < 0) return true;
    char c;
    const ssize_t r = ::recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    if (r >= 0) return false;                       // data pending or clean EOF: not "gone"
    return errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
}

bool Conn::put(const void * p, size_t n) {
    if (dead_) return false;
    if (n == 0) return true;
    if (!send_all(fd_, (const char *) p, n)) { dead_ = true; return false; }
    return true;
}

bool Conn::putf(const char * fmt, ...) {
    if (dead_) return false;
    char stackbuf[4096];
    va_list ap;
    va_start(ap, fmt);
    const int need = std::vsnprintf(stackbuf, sizeof(stackbuf), fmt, ap);
    va_end(ap);
    if (need < 0) { dead_ = true; return false; }
    if ((size_t) need < sizeof(stackbuf)) return put(stackbuf, (size_t) need);
    std::string big;
    big.resize((size_t) need + 1);
    va_start(ap, fmt);
    std::vsnprintf(&big[0], big.size(), fmt, ap);
    va_end(ap);
    big.resize((size_t) need);
    return put(big);
}

const std::string * Request::header(const char * lower_name) const {
    for (const auto & h : headers) {
        if (h.first == lower_name) return &h.second;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// request reading
// ---------------------------------------------------------------------------
static std::string lower_ascii(const std::string & s) {
    std::string o = s;
    for (char & c : o) {
        if (c >= 'A' && c <= 'Z') c = (char) (c - 'A' + 'a');
    }
    return o;
}

static bool parse_content_length(const std::string & v, long long & out) {
    if (v.empty() || v.size() > 19) return false;
    long long n = 0;
    for (char c : v) {
        if (c < '0' || c > '9') return false;
        n = n * 10 + (c - '0');
    }
    out = n;
    return true;
}

// Reads exactly one request off fd. On a protocol failure fills status/message and returns
// false; the caller answers and closes. A CLIENT-caused failure must never be a crash: this
// function is the front line for the prefix fuzz (plan §G4).
static bool read_request(int fd, Request & rq, int & status, std::string & message) {
    std::string buf;
    size_t hdr_end = std::string::npos;   // index of the first body byte
    char chunk[4096];
    while (hdr_end == std::string::npos) {
        if (buf.size() > kMaxHeaderBytes) {
            status  = 413;
            message = "request headers exceed the 8 KiB header cap";
            return false;
        }
        const ssize_t r = ::recv(fd, chunk, sizeof(chunk), 0);
        if (r > 0) {
            buf.append(chunk, (size_t) r);
            size_t p = buf.find("\r\n\r\n");
            if (p != std::string::npos) { hdr_end = p + 4; break; }
            p = buf.find("\n\n");  // tolerate a bare-LF client
            if (p != std::string::npos) { hdr_end = p + 2; break; }
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        status  = (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) ? 408 : 400;
        message = (status == 408) ? "timed out reading the request" : "connection closed before a complete request header";
        return false;
    }

    if (hdr_end > kMaxHeaderBytes) {
        status  = 413;
        message = "request headers exceed the 8 KiB header cap";
        return false;
    }
    const std::string head = buf.substr(0, hdr_end);
    size_t pos = 0;
    // ---- request line ----
    size_t eol = head.find('\n');
    std::string line = head.substr(0, eol == std::string::npos ? head.size() : eol);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    pos = (eol == std::string::npos) ? head.size() : eol + 1;
    {
        const size_t s1 = line.find(' ');
        const size_t s2 = (s1 == std::string::npos) ? std::string::npos : line.find(' ', s1 + 1);
        if (s1 == std::string::npos || s2 == std::string::npos || line.find(' ', s2 + 1) != std::string::npos) {
            status  = 400;
            message = "malformed request line (expected 'METHOD TARGET HTTP/1.x')";
            return false;
        }
        rq.method  = line.substr(0, s1);
        const std::string target = line.substr(s1 + 1, s2 - s1 - 1);
        rq.version = line.substr(s2 + 1);
        if (rq.version.compare(0, 7, "HTTP/1.") != 0) {
            status  = 400;
            message = "unsupported HTTP version (this server speaks HTTP/1.0 and HTTP/1.1)";
            return false;
        }
        const size_t q = target.find('?');
        rq.path  = (q == std::string::npos) ? target : target.substr(0, q);
        rq.query = (q == std::string::npos) ? std::string() : target.substr(q + 1);
        if (rq.path.empty() || rq.path[0] != '/') {
            status  = 400;
            message = "malformed request target (expected an absolute path)";
            return false;
        }
    }
    // ---- headers ----
    bool have_cl = false;
    long long cl = 0;
    while (pos < head.size()) {
        size_t e = head.find('\n', pos);
        std::string h = head.substr(pos, (e == std::string::npos ? head.size() : e) - pos);
        pos = (e == std::string::npos) ? head.size() : e + 1;
        if (!h.empty() && h.back() == '\r') h.pop_back();
        if (h.empty()) continue;
        const size_t c = h.find(':');
        if (c == std::string::npos) {
            status  = 400;
            message = "malformed header line (no ':')";
            return false;
        }
        std::string name = lower_ascii(h.substr(0, c));
        std::string val  = h.substr(c + 1);
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.erase(val.begin());
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
        if (name.empty()) {
            status  = 400;
            message = "malformed header line (empty name)";
            return false;
        }
        if (name == "content-length") {
            long long v = 0;
            if (!parse_content_length(val, v)) {
                status  = 400;
                message = "malformed Content-Length (must be a non-negative decimal integer)";
                return false;
            }
            if (have_cl && v != cl) {
                status  = 400;
                message = "conflicting Content-Length headers";
                return false;
            }
            have_cl = true;
            cl = v;
        }
        if (name == "transfer-encoding") {
            status  = 400;
            message = "Transfer-Encoding is not supported (send Content-Length: no chunked bodies)";
            return false;
        }
        rq.headers.emplace_back(std::move(name), std::move(val));
    }
    // ---- body ----
    const bool body_method = (rq.method == "POST" || rq.method == "PUT" || rq.method == "PATCH");
    if (have_cl && cl > (long long) kMaxBodyBytes) {
        status  = 413;
        message = "request body exceeds the 4 MiB body cap";
        return false;
    }
    if (body_method && !have_cl) {
        status  = 400;
        message = "missing Content-Length on a body-carrying method";
        return false;
    }
    if (have_cl && cl > 0) {
        rq.body.reserve((size_t) cl);
        if (hdr_end < buf.size()) rq.body.append(buf, hdr_end, buf.size() - hdr_end);
        while (rq.body.size() < (size_t) cl) {
            const ssize_t r = ::recv(fd, chunk, sizeof(chunk), 0);
            if (r > 0) { rq.body.append(chunk, (size_t) r); continue; }
            if (r < 0 && errno == EINTR) continue;
            status  = (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) ? 408 : 400;
            message = (status == 408) ? "timed out reading the request body"
                                      : "connection closed before the whole body arrived";
            rq.body.clear();
            return false;
        }
        if (rq.body.size() > (size_t) cl) rq.body.resize((size_t) cl);
    }
    return true;
}

static void send_error(int fd, int status, const std::string & message) {
    const std::string body = error_body("invalid_request_error", message.c_str(), nullptr, nullptr);
    char head[512];
    const int n = std::snprintf(head, sizeof(head),
                                "HTTP/1.1 %d %s\r\nServer: aubier-m1-server/0.1\r\n"
                                "Content-Type: application/json\r\nContent-Length: %zu\r\n"
                                "Connection: close\r\n\r\n",
                                status, reason(status), body.size());
    if (n <= 0) return;
    if (!send_all(fd, head, (size_t) n)) return;
    send_all(fd, body.data(), body.size());
}

// ---------------------------------------------------------------------------
// per-connection worker
// ---------------------------------------------------------------------------
struct Worker {
    int                                   fd = -1;
    const std::function<void(const Request &, Conn &)> * handler = nullptr;

    static void bind(Conn & c, int fd) { c.fd_ = fd; }

    static void * entry(void * arg) {
        Worker * w = (Worker *) arg;
        Request rq;
        int status = 0;
        std::string msg;
        if (!read_request(w->fd, rq, status, msg)) {
            send_error(w->fd, status, msg);
        } else {
            Conn c;
            bind(c, w->fd);
            try {
                (*w->handler)(rq, c);
            } catch (const std::bad_alloc &) {
                if (c.alive())
                    send_error(w->fd, 500, "out of memory while building the response");
            } catch (...) {
                if (c.alive())
                    send_error(w->fd, 500, "internal error while handling the request");
            }
        }
        ::close(w->fd);
        delete w;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            g_inflight.fetch_sub(1, std::memory_order_relaxed);
        }
        g_cv.notify_all();
        return nullptr;
    }
};

// ---------------------------------------------------------------------------
// signals
// ---------------------------------------------------------------------------
static void on_stop_signal(int) {
    const int saved = errno;
    g_stop.store(true, std::memory_order_relaxed);
    if (g_sigpipe[1] >= 0) {
        const char b = 'x';
        const ssize_t ignored = ::write(g_sigpipe[1], &b, 1);
        (void) ignored;
    }
    errno = saved;
}

static bool install_signals() {
    if (::pipe(g_sigpipe) != 0) return false;
    for (int k = 0; k < 2; ++k) {
        const int fl = ::fcntl(g_sigpipe[k], F_GETFL, 0);
        if (fl >= 0) ::fcntl(g_sigpipe[k], F_SETFL, fl | O_NONBLOCK);
    }
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_stop_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (::sigaction(SIGINT, &sa, nullptr) != 0) return false;
    if (::sigaction(SIGTERM, &sa, nullptr) != 0) return false;
    // A client that disappears mid-response must never kill the process (plan §G2).
    struct sigaction ign;
    std::memset(&ign, 0, sizeof(ign));
    ign.sa_handler = SIG_IGN;
    sigemptyset(&ign.sa_mask);
    if (::sigaction(SIGPIPE, &ign, nullptr) != 0) return false;
    return true;
}

// ---------------------------------------------------------------------------
// serve(): the accept loop
// ---------------------------------------------------------------------------
int serve(const Config & cfg, const std::function<void(const Request &, Conn &)> & handler,
          int * out_port) {
    if (!install_signals()) {
        std::fprintf(stderr, "m1_http: FATAL: cannot install signal handlers\n");
        return 1;
    }
    const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        std::fprintf(stderr, "m1_http: FATAL: socket(): %s\n", std::strerror(errno));
        return 1;
    }
    int one = 1;
    ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port   = htons((uint16_t) cfg.port);
    if (::inet_pton(AF_INET, cfg.bind_addr, &sa.sin_addr) != 1) {
        std::fprintf(stderr, "m1_http: FATAL: bad bind address '%s'\n", cfg.bind_addr);
        return 1;
    }
    if (::bind(lfd, (struct sockaddr *) &sa, sizeof(sa)) != 0) {
        std::fprintf(stderr, "m1_http: FATAL: bind(%s:%d): %s\n", cfg.bind_addr, cfg.port,
                     std::strerror(errno));
        return 1;
    }
    if (::listen(lfd, 128) != 0) {
        std::fprintf(stderr, "m1_http: FATAL: listen(): %s\n", std::strerror(errno));
        return 1;
    }
    struct sockaddr_in bound;
    socklen_t blen = sizeof(bound);
    int bound_port = -1;
    if (::getsockname(lfd, (struct sockaddr *) &bound, &blen) == 0)
        bound_port = (int) ntohs(bound.sin_port);
    if (out_port) *out_port = bound_port;
    if (cfg.on_listen) cfg.on_listen(bound_port);

    while (!g_stop.load(std::memory_order_relaxed)) {
        struct pollfd pfd[2];
        pfd[0].fd     = lfd;
        pfd[0].events = POLLIN;
        pfd[0].revents = 0;
        pfd[1].fd     = g_sigpipe[0];
        pfd[1].events = POLLIN;
        pfd[1].revents = 0;
        const int pr = ::poll(pfd, 2, -1);
        if (pr < 0) {
            if (errno == EINTR) continue;
            std::fprintf(stderr, "m1_http: poll(): %s\n", std::strerror(errno));
            break;
        }
        if (pfd[1].revents & POLLIN) break;  // SIGHUP-free: the stop flag is set
        if (!(pfd[0].revents & POLLIN)) continue;

        const int cfd = ::accept(lfd, nullptr, nullptr);
        if (cfd < 0) {
            if (errno == EINTR || errno == ECONNABORTED) continue;
            std::fprintf(stderr, "m1_http: accept(): %s\n", std::strerror(errno));
            continue;
        }
        if (g_stop.load(std::memory_order_relaxed)) {
            send_error(cfd, 503, "server is shutting down");
            ::close(cfd);
            continue;
        }
        // The counter is checked and incremented in the ACCEPT thread, so --max-conns is a
        // hard bound and no thread is created for a refused connection.
        int cur = g_inflight.load(std::memory_order_relaxed);
        while (cur < cfg.max_conns &&
               !g_inflight.compare_exchange_weak(cur, cur + 1, std::memory_order_relaxed)) {
        }
        if (cur >= cfg.max_conns) {
            const std::string body =
                cfg.busy_body.empty()
                    ? error_body("server_error", "server busy: too many in-flight connections",
                                 nullptr, "server_busy")
                    : cfg.busy_body;
            char head[512];
            const int n = std::snprintf(head, sizeof(head),
                                        "HTTP/1.1 503 Service Unavailable\r\n"
                                        "Server: aubier-m1-server/0.1\r\nRetry-After: 1\r\n"
                                        "Content-Type: application/json\r\nContent-Length: %zu\r\n"
                                        "Connection: close\r\n\r\n",
                                        body.size());
            if (n > 0 && send_all(cfd, head, (size_t) n)) send_all(cfd, body.data(), body.size());
            ::close(cfd);
            continue;
        }
        // socket options for a served connection
        struct timeval tv;
        tv.tv_sec  = cfg.read_timeout_ms / 1000;
        tv.tv_usec = (cfg.read_timeout_ms % 1000) * 1000;
        ::setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        Worker * w = new Worker();
        w->fd      = cfd;
        w->handler = &handler;
        pthread_t th;
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
        const int rc = pthread_create(&th, &at, &Worker::entry, w);
        pthread_attr_destroy(&at);
        if (rc != 0) {
            std::fprintf(stderr, "m1_http: pthread_create: %s\n", std::strerror(rc));
            delete w;
            {
                std::lock_guard<std::mutex> lk(g_mu);
                g_inflight.fetch_sub(1, std::memory_order_relaxed);
            }
            g_cv.notify_all();
            send_error(cfd, 503, "server busy: cannot start a worker thread");
            ::close(cfd);
        }
    }

    ::close(lfd);
    // stop accepting is one-way; now let in-flight handlers observe stopping() and abort at the
    // next position boundary. Bounded: a wedged client must not hold the process open forever.
    g_stop.store(true, std::memory_order_relaxed);
    {
        std::unique_lock<std::mutex> lk(g_mu);
        g_cv.wait_for(lk, std::chrono::milliseconds(cfg.drain_ms),
                      [] { return g_inflight.load(std::memory_order_relaxed) == 0; });
    }
    return 0;
}

}  // namespace m1http
