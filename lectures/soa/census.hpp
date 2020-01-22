#ifndef CSC586C_CENSUS
#define CSC586C_CENSUS


#include <array>
#include <random>

namespace csc586 {

enum class Sex { male, female };
enum class HousingStatus { renter, owner };

using Age = uint8_t;
using Income = uint32_t;
using Ethnicity = uint16_t;
using Language = uint16_t;
using Religion = uint16_t;
using Country = uint8_t;

namespace distributions {

using sex_distr     = std::uniform_int_distribution< uint32_t >;
using country_distr = std::uniform_int_distribution< Country >;
using age_distr     = std::binomial_distribution< Age >;
using hst_distr     = std::bernoulli_distribution;
using income_distr  = std::geometric_distribution< Income >;

} // namespace distributions

namespace aos {

struct person
{
	Age age;
	Country country;
	Income income;
	HousingStatus hst;
	Sex sex;
};

struct build_person
{
	person operator() ()
	{
		using namespace csc586::distributions;

		return person {
			  age_distr( 120, .25 )( generator )
			, country_distr( 0, 255 )( generator )
			, income_distr( .5 )( generator ) + 10000
			, hst_distr( 0.68 )( generator ) ? HousingStatus::owner : HousingStatus::renter
			, sex_distr( 0, 1 )( generator ) == 0 ? Sex::male : Sex::female
		};
	}

private:
	std::default_random_engine generator;
};


using census = std::vector< person >;


census create_random_census( size_t const population_size )
{
	census population_data( population_size );
	std::generate( population_data.begin(), population_data.end(), build_person{} );
	return population_data;
}

template < size_t NUM_BUCKETS >
	auto bucketise_by_age( census const& population_data )
	{
		std::array< size_t, NUM_BUCKETS > buckets{};
		for( auto const& person : population_data )
		{
			++buckets[ person.age ];
		}
		return buckets;
	}

} // namespace aos






namespace soa {

struct census
{
	std::vector< Sex > sex;
	std::vector< Age > age;
	std::vector< Income > income;
	std::vector< Country > country;
	std::vector< HousingStatus > hst;
};


census create_random_census( size_t const population_size )
{
	using namespace csc586::distributions;

	std::default_random_engine generator;

	std::vector< Sex > sex( population_size );
	std::vector< Age > age( population_size );
	std::vector< Income > income( population_size );
	std::vector< Country > country( population_size );
	std::vector< HousingStatus > hst( population_size );

	std::generate( sex.begin()
				 , sex.end()
				 , [ &generator ](){ return sex_distr(0,1)( generator ) == 0
				 								? Sex::male
				 								: Sex::female; } );

	std::generate( age.begin()
				 , age.end()
				 , [ &generator ](){ return age_distr( 120, .25 )( generator ); } );

	std::generate( income.begin()
				 , income.end()
				 , [ &generator ](){ return income_distr( 0.5 )( generator ); } );

	std::generate( country.begin()
				 , country.end()
				 , [ &generator ](){ return country_distr( 0, 255 )( generator ); } );

	std::generate( hst.begin()
				 , hst.end()
				 , [ &generator ](){ return hst_distr( 0.68 )( generator ) 
				 								? HousingStatus::owner
				 								: HousingStatus::renter; } );

	return census{ sex, age, income, country, hst };
}

template < size_t NUM_BUCKETS >
	auto bucketise_by_age( census const& population_data )
	{
		std::array< size_t, NUM_BUCKETS > buckets{};
		for( auto const age : population_data.age )
		{
			++buckets[ age ];
		}
		return buckets;
	}

} // namespace soa
} // namespace csc586

#endif // CSC586C_CENSUS