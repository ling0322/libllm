set(lutil_SOURCES
    "internal/log.cc"
    "internal/sprintf.cc"
    "base64.cc"
    "error.cc"
    "flags.cc"
    "half.cc"
    "ini_config.cc"
    "is_debug.cc"
    "path.cc"
    "random.cc"
    "reader.cc"
    "strings.cc"
    "time.cc"
    "thread_pool.cc"
    "zip_file.cc")

set(lut_test_SOURCES
    "path_test.cc"
    "strings_test.cc")

# OS specific code
if(WIN32)
    set(lutil_SOURCES
        ${lutil_SOURCES}
        "path_windows.cc"
        "platform_windows.cc"
        "shared_library_windows.cc")
endif()
if(UNIX)
    set(lutil_SOURCES
        ${lutil_SOURCES}
        "platform_linux.cc"
        "shared_library_linux.cc")
endif()
if(UNIX AND APPLE)
    set(lutil_SOURCES ${lutil_SOURCES} "path_darwin.cc")
endif()
if(UNIX AND NOT APPLE)
    set(lutil_SOURCES ${lutil_SOURCES} "path_linux.cc")
endif()

add_library(lutil STATIC ${lutil_SOURCES})
set_target_properties(lutil PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(lutil PRIVATE ".." ${LIBDWARF_INCLUDE_DIR})

add_library(lutil_test OBJECT ${lut_test_SOURCES})
target_include_directories(lutil_test PRIVATE "..")
