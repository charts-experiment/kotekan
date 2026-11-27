#ifndef KOTEKAN_LOGGING_HPP
#define KOTEKAN_LOGGING_HPP

#include <cstdio>
#include <cstdlib>

namespace kotekan {
struct kotekanLogging {
    template <typename... Args>
    static void internal_logging(int, const char*, const char*, Args&&...) {}
};
} // namespace kotekan

#ifndef INFO_NON_OO
#define INFO_NON_OO(m, ...) do {} while (0)
#endif
#ifndef WARN_NON_OO
#define WARN_NON_OO(m, ...) do {} while (0)
#endif
#ifndef ERROR_NON_OO
#define ERROR_NON_OO(m, ...) do {} while (0)
#endif
#ifndef DEBUG_NON_OO
#define DEBUG_NON_OO(m, ...) do {} while (0)
#endif
#ifndef FATAL_ERROR
#define FATAL_ERROR(m, ...) do { std::fprintf(stderr, m "\n", ##__VA_ARGS__); std::exit(1); } while (0)
#endif

#endif // KOTEKAN_LOGGING_HPP
