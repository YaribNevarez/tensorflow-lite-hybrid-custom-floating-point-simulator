#include "main_functions.h"
#include "stdio.h"
#include "stdlib.h"
#ifdef __linux__    // Linux only
#include <sched.h>  // sched_setaffinity
#endif

int main (int argc, char* argv[])
{
//#ifdef __linux__
//    int cpuAffinity = 2;//argc > 1 ? atoi(argv[1]) : -1;
//
//    if (cpuAffinity > -1)
//    {
//        cpu_set_t mask;
//        int status;
//
//        CPU_ZERO(&mask);
//        CPU_SET(cpuAffinity, &mask);
//        status = sched_setaffinity(0, sizeof(mask), &mask);
//        if (status != 0)
//        {
//            printf("sched_setaffinity");
//        }
//    }
//#endif
  setup ();
  while (true)
  {
    loop ();
  }
}
