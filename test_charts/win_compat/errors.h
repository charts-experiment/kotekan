#ifndef ERRORS
#define ERRORS

#include "syslog.h"
#include <cstdio>
#include <cstdlib>
#include <cerrno>

enum ReturnCode {
    CLEAN_EXIT = 0,
    FATAL_ERROR = 1,
    TEST_PASSED,
    TEST_FAILED,
    RETURN_CODE_COUNT,
    DATASET_MANAGER_FAILURE
};

#define DEBUG_F(m, ...) do {} while (0)
#define DEBUG2_F(m, ...) do {} while (0)
#define ERROR_F(m, ...) do {} while (0)
#define WARN_F(m, ...) do {} while (0)
#define INFO_F(m, ...) do {} while (0)
#define TEST_PASSED() std::exit(0)
#define TEST_FAILED() std::exit(1)

#endif
