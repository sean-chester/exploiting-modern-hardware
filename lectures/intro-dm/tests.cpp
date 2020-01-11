/**
 * This file generates a unit test suite using the Catch2 header-only library.
 * The library contains a main method to make this a stand alone application,
 * which consists of all our tests.
 * 
 * A unit test is a test that isolates and verifies an isolated unit of an application.
 * In this lecture, the solutions are all very small, so we're testing them directly.
 * We will typically take a test-first approach in class to clarify the goals/specs
 * before we start. If you are not thoroughly comfortable with unit testing, I recommend
 * the following resource: https://www.toptal.com/qa/how-to-write-testable-code-and-why-it-matters
 * Catch2 also provides a lot of more advanced tips: https://github.com/catchorg/Catch2/tree/master/docs
 *
 * I have selected Catch2 because of its simplicity. (It is the unit test framework that I use in
 * my own research as well.) Alternatives such as GTest by Google are more ubiquitous. The trade-off
 * for Catch's simplicity is a slow build time; however, this only affects your test suite, not
 * your main application.
 *
 * As the problems that we tackle become more difficult, a solid testing methodology becomes even
 * more important. But now's as good a time as any to become comfortable with both unit testing
 * and the concept of tests-as-documentation.
 */

// The following two verbatim lines are all that are required to use Catch2.
// Note that you may have to adjust the path to catch.hpp in your include statement,
// depending on where you downloaded it to and how you have set your $PATH variables.

#define CATCH_CONFIG_MAIN       // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"            // This includes the catch header, which defines the entire library

#include "unique_element.hpp"  // This includes the file that declares all the code that we want to test


// A "test case" is a grouping of tests based on a common scenario
// You are welcome to define these however makes sense to your application
// Ideally, you should have one test case to cover each branch of control flow that your code may take
// I.e., these tests are *incomplete*
// This is an example of tests-as-documentation. We simultaneously describe how the system is
// supposed to behave *and* test that it behaves that way. If the documentation becomes obsolete,
// so does the test, and then it fails. When writing documentation, it is worth considering
// whether those comments would make for a good test, too/instead.
TEST_CASE( "Vector of only one element", "[unique]" )
{
    // The following lines set up the conditions of the test
    // In this case, a vector (i.e., dynamic array) of exactly one element
	uint32_t const elem = 5u;
	std::vector< uint32_t > one_elem_input{ elem };

    // The following lines are tests/asserts
    // If one fails, the whole test case fails.
    // I have created one test for each solution that we defined in unique_elements.hpp
    REQUIRE( elem == csc586::unique::map_based ( one_elem_input ) );
    REQUIRE( elem == csc586::unique::bit_based ( one_elem_input ) );
    REQUIRE( elem == csc586::unique::two_loops ( one_elem_input ) );
    REQUIRE( elem == csc586::unique::skip_based( one_elem_input ) );
    REQUIRE( elem == csc586::unique::sort_based( one_elem_input ) );
}


/**
 * Simple test case where the unique element is sorted to the back of the array
 */
TEST_CASE( "Vector of only one non-unique element", "[unique]" )
{
	uint32_t const elem = 5u;
	std::vector< uint32_t > two_elem_input{ elem / 2, elem, elem / 2 };

    // In Catch2, a check assertion will not break the test if it fails.
    // It will continue to check all the other tests until a REQUIRE fails.
    CHECK( elem == csc586::unique::map_based ( two_elem_input ) );
    CHECK( elem == csc586::unique::bit_based ( two_elem_input ) );
    CHECK( elem == csc586::unique::two_loops ( two_elem_input ) );
    CHECK( elem == csc586::unique::skip_based( two_elem_input ) );
    CHECK( elem == csc586::unique::sort_based( two_elem_input ) );
}

/**
 * Tests behaviour when a repeated element appears an odd number of times (breaking
 * the xor algorithm, for example)
 */
TEST_CASE( "Vector contains a triplet", "[unique]" ) {
	uint32_t const elem = 5u;
	std::vector< uint32_t > triplet_input{ elem / 2, elem, elem / 2, elem / 2 };

    REQUIRE( elem == csc586::unique::map_based ( triplet_input ) );
    REQUIRE( elem == csc586::unique::two_loops ( triplet_input ) );
    REQUIRE( elem == csc586::unique::sort_based( triplet_input ) );

    // In this case, we confirm that the xor algorithm and the skipping sort-based
    // algorithms fail on this input.
    // If it were to pass, that would represent a bug in the code.
    // It is important to test negative instances, as well, not just
    // positive ones (i.e., to achieve full branch coverage)
    REQUIRE_FALSE( elem == csc586::unique::bit_based ( triplet_input ) );
    REQUIRE_FALSE( elem == csc586::unique::skip_based( triplet_input ) );
}
