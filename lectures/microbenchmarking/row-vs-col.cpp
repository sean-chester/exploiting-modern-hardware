/**
 * A reproduction of the experiment in Doumler (2016) that shows the difference between walking in
 * a row-oriented versus column-oriented fashion through a two-dimensional array. You should observe
 * a circa 4x speed-up using row-oriented iterations because:
 *    a) you have better *temporal locality* with cache line reuse
 *    b) you have better *spatial locality* when you read sequentially through data
 *
 * Bonus: Try changing to a random access pattern by picking a random column instead. Relative to a
 * sequential scan, you should observe circa 19x slower performance on a row-oriented scan with random
 * columns and a 30x slower performance on a column-oriented scan!
 *
 * For simplicity the code below uses a linearised 2d table. E.g., a 3x3 table T is laid out in 1d as:
 * [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
 */

#include <iostream>  // std::cout
#include <cassert>	 // assert()

#include "timing.hpp"
#include "data-generation.hpp"


namespace row_oriented
{

struct calc_sum
{
    size_t const num_cols;

	template < typename T >
		T operator () ( std::vector< T > data ) const
		{
			assert( "Not empty" && data.size() > 0 );

			T sum = 0u;
            auto const num_rows = data.size() / num_cols;

            for( auto i = 0u; i < num_rows; ++i )
            {
                for( auto j = 0u; j < num_cols; ++j )
                {
                    // TODO: Try changing `j` to `( rand() % num_cols )` to get random patterns
                    sum += data[ i * num_cols + ( rand() % num_cols ) ]; // cell (i,j) in linearised 2d array
                }
            }

            return sum;
		}
};
}


namespace col_oriented
{

struct calc_sum
{
    size_t const num_cols;

	template < typename T >
		T operator () ( std::vector< T > data ) const
		{
			assert( "Not empty" && data.size() > 0 );

            T sum = 0u;
            auto const num_rows = data.size() / num_cols;

            for( auto j = 0u; j < num_cols; ++j )
            {
                for( auto i = 0u; i < num_rows; ++i )
                {
                    // TODO: Try changing `j` to `( rand() % num_cols )` to get random patterns
                    sum += data[ i * num_cols + j ]; // cell (i,j) in linearised 2d array
                }
            }

            return sum;
		}
};
}


// Observe that we are now taking command line arguments so that we can run this with different options
// rather than recompiling for every test.
int main( int argc, char **argv )
{
	auto num_tests  = 10u; // Number of random trials to test out

    // argc gives the *argument count*, i.e., the number of arguments passed by the user
    // argv gives the *argument values*. argv[0] is always the program name so that you can echo it back to the user.
    // Here I use very lightweight input parameter checking: if the user didn't provide extra arguments,
    // i.e., argc <= 1, then print out the instructions and quite the program
    if( argc <= 1 )
    {
        std::cout << "Usage: " << argv[ 0 ] << " <num_rows> <num_cols> [use col-oriented format]" << std::endl;
        return 0;
    }

    // std::strtoul() converts an ascii STRing TO an Unsigned Long
    // this converts the string input from the user into unsigned long integers to use in the program.
    auto const num_rows = std::strtoul( argv[ 1 ], NULL, 0 );
    auto const num_cols = std::strtoul( argv[ 2 ], NULL, 0 );

    // For random numbers, one must first seed the random number generator. This is the idiomatic
    // approach for the random number generator libraries that we have chosen.
    std::srand ( static_cast< uint32_t >( std::time(0) ) );

    // Note that I have "linearised" the 2d array: i.e., instead of creating a T[][] c-style 2d array
    // like in the Doumler slides, I continue to use a 1d std::vector() and I will just use offsets
    // (e.g., array[ i * width + j]) to index into cell (i,j).
	auto const test_cases = csc586::benchmark::uniform_rand_vec_of_vec< uint32_t >( num_tests
                                                                                  , num_rows * num_cols );
	
	// not elegant, but this switches which implementation to run depending on whether a third argument was given
	auto const run_time   = argc > 3 ? csc586::benchmark::benchmark( col_oriented::calc_sum{ num_cols }
																   , test_cases )
                                     : csc586::benchmark::benchmark( row_oriented::calc_sum{ num_cols }
                                                                   , test_cases );

    std::cout << "Average time (us): " << run_time << std::endl;

	return 0;
}
