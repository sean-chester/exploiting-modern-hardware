/**
 * Timing library to benchmark and comparatively analyse different implementation approaches
 */

#ifndef CS586_TIMING
#define CS586_TIMING

#include <algorithm> // std::for_each()
#include <chrono>	 // timing libraries

namespace csc586 {
namespace benchmark {

using duration = float;

/**
 * Benchmarks the average time to run Function f, using the provided container of test_instances.
 * For more accurate timing results, it is best to provide many independently generated test instances.
 *
 * Observe that this is a generic (i.e., template) function and an example of _generic functional
 * programming_ in c++; i.e., the first argument, variable f, is itself another "callable" functor. The
 * second argument is any type of iterable container (e.g., set, list, vector, array) of test instances. 
 *
 * Many high-level languages allow overloading functions. In C++, you can even overload operators, such
 * as +, -, ~, and ==. There is a specific operator, the function-call operator, denoted by (). If you
 * overload this in a class, then your class can be used as function. However, unlike standard functions,
 * it can have member variables (i.e., state). A class for which the function-call operator has been
 * overloaded is called a "functor". In generic programming, we pass an instance of that callable
 * class as a parameter to the function, but we have to templatise the type, because we want to be
 * able to pass *any* function as an argument. The advantage of this approach, aside from code reusability,
 * is that the compiler can aggressively optimise for each particular functor, as it can be directly
 * inlined at compile time. For example of defining, initialising, and using  functor, see benchmarking.cpp
 * or https://stackoverflow.com/a/317528/2769271
 *
 * Because we have defined this benchmark generically, we can (and will!) reuse it for any function we want.
 */
template < typename Callable, typename Container >
	duration benchmark( Callable f, Container test_instances )
	{
		// One main challenge with generic programming is that we don't know any types! We may want
		// to declare a variable whose type depends on our template parameters (e.g., the type returned. 
		// by f(). The decltype() function does exactly this for us: it returns the type of an expression.
		// In the example below, the expression is to call f on the first test instance; decltype() returns
		// the type of the output of f(). If f() were benchmark(), for example, then decltype() would return
		// duration. In this way, we can bind an alias, output_type, to the not-yet-known return type of f().
		// decltype() is evaluated at compile time.
		// Note that in some cases the auto keyword is enough, and we don't need to use decltype(), but we
		// use it here because we will need to declare an accumulator variable without knowing a suitable
		// initial value.
		using output_type = decltype( f( test_instances.front() ) );

		// Modern c++ compilers are very clever and aggressive with optimisation. If they observe
		// that some code does not contribute side-effects no to the output of a function, they
		// may remove it entirely. That could happen to our entire benchmark loop below, depending on
		// the definition of f()! To ensure that does not happen in our benchmark, we want to use the
		// output value in some manner. In this implementation, we accumulate it in a variable and
		// print it out. This unfortunately restricts our generic programming, as now the output type
		// of f() has to have both the + and the << operators defined. Moreover, it adds a small
		// overhead to our benchmark.
		// Nonetheless, try benchmarking our bit_based() algorithm without this accumulator in place
		// and you'll see the importance of this small overhead!
		output_type output = 0;

		// starts the timer. We use the chrono library not just because it is the idiomatic approach,
		// but also because it offers precision to the nanosecond, if we want it.
		auto const start_time = std::chrono::steady_clock::now();

		// run function f on every random test instance, arbitrarily summing the return values.
		// Note that for_each is a standard library function that applies a functor (the third argument)
		// to everything in the range given by the first two parameters (start and end).
		//
		// cbegin() returns an iterator to the first element in a container (such as a vector or array).
		// In particular, the iterator is *const*; i.e., we cannot modify the contents using the iterator.
		// cend() returns an iterator one past the last element. So looping from cbegin() to cend() will
		// iterate everything in test_instances.
		// 
		// We could have simply passed our functor f to for_each, but we need to modify it so that we do
		// something with the output. We have used a lambda to create this new functor at the point of
		// use. Lambdas are also available in Java; so, you should already have been exposed to them.
		// If not, consult https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp?view=vs-2019 
		// for a technical explanation or https://stackoverflow.com/q/7627098/2769271 for an example-based
		// explanation. In short, the example below creates a one-time-use function with arguments specified
		// in the parentheses (), with definition specified in the braces {}, and using (i.e., "capturing")
		// the variables indicated in the brackets [].
		std::for_each( std::cbegin( test_instances )
					 , std::cend  ( test_instances )
					 , [&output, f]( auto const& v ){ output = output + f(v); } );

		// end timer
		auto const end_time = std::chrono::steady_clock::now();

		// do something arbitrary with output. In this case, we print it out.
		std::cout << output << std::endl;

		// return average time
		// the syntax for this library is a bit cumbersome...
		return std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count()
			/ static_cast< duration >( test_instances.size() );
	}


} // namespace benchmark
} // namespace csc586

#endif // CS586_TIMING