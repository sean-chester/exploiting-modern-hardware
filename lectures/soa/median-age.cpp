#include <iostream>
#include <array>
#include <ctime> // for initialising random seed
#include <cassert>
#include <algorithm> // std::generate
#include <chrono>	 // timing libraries

#include "census.hpp"


template < typename T >
	size_t find_median_bucket( T buckets, size_t const n )
	{
		assert( "n matches cumulative total of all buckets"
				&& std::accumulate( buckets.cbegin(), buckets.cend(), 0u ) == n );

		size_t cumulative_total = 0u;

		for( auto i = 0u; i < buckets.size(); ++i )
		{
			cumulative_total += buckets[ i ];
			if( cumulative_total > n / 2u ) { return i; }
		}

		// unreachable if sum of bucket counts == n, as per assert above
		return buckets.size();
	}

int main()
{
	using namespace csc586;
	using namespace csc586::soa;

	std::srand ( static_cast< uint32_t >( std::time(0) ) ); // seed random num generator

	auto const population_size = 100000000u;
	auto const age_bound = 1u << ( 8 * sizeof( Age ) );

	auto const population_data = create_random_census( population_size );
	auto const start_time = std::chrono::steady_clock::now();

	auto const median_age = find_median_bucket( bucketise_by_age< age_bound >( population_data )
											  , population_size );


	auto const end_time = std::chrono::steady_clock::now();
	std::cout << "Calculation time = "
			  << std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count()
			  << std::endl;
	std::cout << "Median age = " << median_age << std::endl;
	return 0;
}