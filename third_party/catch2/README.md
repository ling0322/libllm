# Catch2

Vendored [Catch2](https://github.com/catchorg/Catch2) v3.3.2, as the two-file amalgamated
distribution (`catch_amalgamated.hpp` and `catch_amalgamated.cpp`), generated 2023-02-26.
Distributed under the Boost Software License 1.0.

## Local modifications

The amalgamated files say "You probably shouldn't edit it directly", so anything changed here is
listed below and has to be re-applied when the version is bumped.

### `catch_amalgamated.hpp`: default to `CATCH_CONFIG_PREFIX_ALL`

Right after the include guard:

```cpp
#if !defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_NO_PREFIX_ALL)
#define CATCH_CONFIG_PREFIX_ALL
#endif
```

The tests use the prefixed macro names (`CATCH_TEST_CASE`, `CATCH_REQUIRE`, ...), which upstream
only enables when `CATCH_CONFIG_PREFIX_ALL` is defined by the build. The top-level `CMakeLists.txt`
does pass `-DCATCH_CONFIG_PREFIX_ALL`, but a translation unit parsed outside the build system does
not get it, so an editor reading a test file on its own reports every `CATCH_TEST_CASE` as
undefined. Defining it in the header keeps the header self-contained.

The `!defined(CATCH_CONFIG_PREFIX_ALL)` guard matters: the command line defines the macro as `1`
while this defines it as empty, and without the guard that is a macro redefinition. The build flag
is now redundant, but it is kept so that re-vendoring the header without this patch breaks loudly
at compile time instead of silently switching every test to the unprefixed spelling.

`CATCH_CONFIG_NO_PREFIX_ALL` opts back out.

## Upgrading

Download the amalgamated files for the new release from the Catch2 repository, overwrite both files,
then re-apply the modification above.
