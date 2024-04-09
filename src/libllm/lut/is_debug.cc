// original file: https://github.com/Tencent/ncnn/blob/056509a/src/cpu.cpp

#include "libllm/lut/is_debug.h"

#include <stdint.h>

#if defined _WIN32
#include <windows.h>
#endif

#if defined __ANDROID__ || defined __linux__
#include <unistd.h>
#endif

#if defined __APPLE__
#include <sys/sysctl.h>
#include <unistd.h>
#endif

namespace lut {

bool isDebug()
{
#if defined _WIN32
    return IsDebuggerPresent();
#elif defined __ANDROID__ || defined __linux__
    // https://stackoverflow.com/questions/3596781/how-to-detect-if-the-current-process-is-being-run-by-gdb
    int status_fd = open("/proc/self/status", O_RDONLY);
    if (status_fd == -1)
        return false;

    char buf[4096];
    ssize_t num_read = read(status_fd, buf, sizeof(buf) - 1);
    close(status_fd);

    if (num_read <= 0)
        return false;

    buf[num_read] = '\0';
    const char tracerPidString[] = "TracerPid:";
    const char* tracer_pid_ptr = strstr(buf, tracerPidString);
    if (!tracer_pid_ptr)
        return false;

    for (const char* ch = tracer_pid_ptr + sizeof(tracerPidString) - 1; ch <= buf + num_read; ++ch)
    {
        if (isspace(*ch))
            continue;

        return isdigit(*ch) != 0 && *ch != '0';
    }

    return false;
#elif defined __APPLE__
    // https://stackoverflow.com/questions/2200277/detecting-debugger-on-mac-os-x
    struct kinfo_proc info;
    info.kp_proc.p_flag = 0;

    int mib[4];
    mib[0] = CTL_KERN;
    mib[1] = KERN_PROC;
    mib[2] = KERN_PROC_PID;
    mib[3] = getpid();

    size_t size = sizeof(info);
    sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &size, NULL, 0);

    return ((info.kp_proc.p_flag & P_TRACED) != 0);
#else
    // unknown platform :(
    fprintf(stderr, "unknown platform!\n");
    return false;
#endif
}

}  // namespace lut