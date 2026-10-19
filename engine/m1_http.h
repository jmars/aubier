// engine/m1_http.h — the server's HTTP/1.1 plumbing: accept loop, per-connection pthreads,
// request parsing with hard caps, connection-close responses, graceful shutdown.
//
// Server plan §F, unit U3. POSIX sockets + pthreads ONLY (MEASURED: no boost/asio, no
// OpenSSL on the box, and none may be added). Deliberate simplifications, all of them
// one-way-safe:
//
//   * Connection: close on EVERY response — no keep-alive, no pipelining. A connection carries
//     exactly one request, so there is no idle-connection state to get wrong.
//   * header block capped at 8 KiB, body at 4 MiB. Over-size is answered 413 BEFORE the body
//     is read, then the connection is closed. A request that is exactly at the cap is fine.
//   * Content-Length is REQUIRED on a body-carrying method; Transfer-Encoding (chunked) is
//     refused with 400, because we do not implement de-chunking and a half-read body would
//     silently truncate a prompt.
//   * one DETACHED pthread per connection, with an in-flight counter checked in the accept
//     loop => 503 (with Retry-After) beyond --max-conns. The counter is what bounds the server
//     under a flood: no thread is created for a refused connection.
//   * SIGPIPE is ignored process-wide; a write to a dead client returns false and sets the
//     connection dead, and the handler's cancel callback sees it at the next position
//     boundary (plan §E).
//   * SIGINT/SIGTERM sets a stop flag and writes a self-pipe, so the accept loop wakes even
//     when no connection arrives. The server then stops accepting, lets in-flight handlers
//     observe stopping() and abort at a position boundary, and waits a bounded time for the
//     counter to drain.
//
// The 2xx/4xx JSON bodies are the ROUTER's business (engine/m1_server.cpp). This layer only
// needs a shape for its own refusals (413, 400 malformed HTTP) and for the 503 refusal, which
// is why error_body() lives here and is public.

#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace m1http {

static const size_t kMaxHeaderBytes = 8u * 1024u;      // the whole header block, incl. the CRLFCRLF
static const size_t kMaxBodyBytes   = 4u * 1024u * 1024u;

struct Request {
    std::string method;   // verbatim, as the client sent it
    std::string path;     // target up to '?', verbatim (no percent-decoding: no route needs it)
    std::string query;    // target after '?', verbatim (may be empty)
    std::string version;  // "HTTP/1.1" / "HTTP/1.0"
    std::vector<std::pair<std::string, std::string>> headers;  // names LOWERCASED, values trimmed
    std::string body;

    const std::string * header(const char * lower_name) const;
};

// One connection's write side. Every put() returns false once the peer is gone; once that has
// happened the connection stays dead, so a streaming handler's loop condition is just !alive().
class Conn {
  public:
    bool head(int status, const char * content_type, const char * extra_headers,
              long long content_length);  // content_length < 0 => close-delimited body (SSE)
    bool put(const void * p, size_t n);
    bool put(const std::string & s) { return put(s.data(), s.size()); }
    // printf-style append into one write() call (SSE chunks are built this way)
    bool putf(const char * fmt, ...) __attribute__((format(printf, 2, 3)));

    bool alive() const { return !dead_; }
    void kill() { dead_ = true; }
    int fd() const { return fd_; }

  private:
    friend struct Worker;
    int  fd_   = -1;
    bool dead_ = false;
};

// True once SIGINT/SIGTERM has been received. A streaming handler polls it (through the
// engine's cancel callback) and aborts at the next position boundary.
bool stopping();

// The OpenAI-shaped error body, one definition for the whole server:
//   {"error":{"message":..,"type":..,"param":..,"code":..}}
// param/code/type may be null (passed as nullptr). `extra` is appended INSIDE the error object,
// already formatted as JSON members without a leading comma (or nullptr).
std::string error_body(const char * type, const char * message, const char * param,
                       const char * code, const char * extra = nullptr);

struct Config {
    const char * bind_addr      = "127.0.0.1";
    int          port           = 0;      // 0 => ephemeral; the chosen port is printed
    int          max_conns      = 32;      // in-flight connection cap
    int          read_timeout_ms = 30000;  // per-socket SO_RCVTIMEO
    int          drain_ms       = 10000;   // how long shutdown waits for in-flight handlers
    std::string  busy_body;               // the 503 body for over-cap and shutdown refusals
    // Called once, AFTER bind() and BEFORE the first accept, with the port actually bound
    // (--port 0 => ephemeral). This is where the server prints/records the chosen port.
    std::function<void(int)> on_listen;
};

// Runs until SIGINT/SIGTERM. Returns the process exit code (0 clean, 1 if the socket could not
// be set up). The chosen port is reported through out_port.
int serve(const Config & cfg, const std::function<void(const Request &, Conn &)> & handler,
          int * out_port);

}  // namespace m1http
