#include <gtest/gtest.h>

#include "testFixtures/MpiTestEventListener.h"
#include "testFixtures/MpiTestFixture.hpp"

//int main2(int argc, char** argv);
//
//int main(int argc, char** argv) {
//    return main2(argc, argv);
//    ::testing::InitGoogleTest(&argc, argv);
//
//    // Store the input parameters
//    const bool inMpiTestRun = MpiTestFixture::InitializeTestingEnvironment(&argc, &argv);
//    if (inMpiTestRun) {
//        testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
//        delete listeners.Release(listeners.default_result_printer());
//
//        listeners.Append(new MpiTestEventListener());
//    }
//
//    return RUN_ALL_TESTS();
//}