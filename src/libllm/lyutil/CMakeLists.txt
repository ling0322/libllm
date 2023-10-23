set(lyutil_SOURCES
    "internal/log.cc"
    "internal/sprintf.cc"
    "error.cc"
    "flags.cc"
    "ini_config.cc"
    "path.cc"
    "random.cc"
    "reader.cc"
    "strings.cc"
    "time.cc")

if(WIN32)
    set(lyutil_SOURCES
        ${lyutil_SOURCES}
        "path_windows.cc"
        "platform_windows.cc"
        "shared_library_windows.cc")
endif(WIN32)
if(UNIX)
    set(lyutil_SOURCES
        ${lyutil_SOURCES}
        "path_linux.cc"
        "platform_linux.cc")
endif(UNIX)

add_library(lyutil STATIC ${lyutil_SOURCES})
target_include_directories(lyutil PRIVATE .. ../..)
target_compile_features(lyutil PRIVATE cxx_std_14)
if(UNIX)
  target_link_libraries(lyutil -ldl)
endif(UNIX)

set(lyutil_test_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/strings_test.cc"
    "${CMAKE_CURRENT_LIST_DIR}/path_test.cc" PARENT_SCOPE)
