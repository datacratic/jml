/** watchdog.h                                                     -*- C++ -*-
    Jeremy Barnes, 16 May 2011
    Copyright (c) 2011 Datacratic.  All rights reserved.

    Watchdog timer class.
*/

#ifndef __jml_testing__watchdog_h__
#define __jml_testing__watchdog_h__

#include <signal.h>
#include <functional>
#include <thread>
#include <iostream>

namespace ML {

struct Watchdog {
    bool finished;
    double seconds;
    std::function<void ()> timeoutFunction;
    std::thread wdThread;
    
    static void abortProcess()
    {
        using namespace std;

        cerr << "**** WATCHDOG TIMEOUT; KILLING HUNG TEST ****"
             << endl;
        abort();
        kill(0, SIGKILL);
    }
    
    void runThread()
    {
        struct timespec ts = { 0, 10000000 };
        struct timespec rem;
        for (unsigned i = 0;  i != int(seconds * 100) && !finished;
             ++i) {
            nanosleep(&ts, &rem);
        }
        
        if (!finished)
            timeoutFunction();
    }
    
    /** Create a watchdog timer that will time out after the given number
        of seconds.
    */
    Watchdog(double seconds = 2.0,
             std::function<void ()> timeoutFunction = abortProcess)
        : finished(false), seconds(seconds), timeoutFunction(timeoutFunction),
          wdThread(std::bind(&Watchdog::runThread, this))
    {
    }

    ~Watchdog()
    {
        finished = true;
        wdThread.join();
    }
};

} // namespace ML

#endif /* __jml_testing__watchdog_h__ */

