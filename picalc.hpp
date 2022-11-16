/**
 * @file picalc.hpp
 * @author Benjamin Dreyer
 * @date 2022-11-16
 *
 * This file contains implementations for the Dimension Specific Algorithm and
 * the Transdimensional Algorithm which are two algorithms conjectured to
 * calculate digits of pi efficiently. These implementations shall serve as
 * illustration for the conjecture.
 * 
 * The file contains a namespace picalc which contains the three namespaces
 * `simple`, `efficient` and `optimized`. Each of these contains two functions,
 * `dimensions_specific(...)` and `trans_dimensional(...)` that correspond to
 * the respective algorithms.
 * 
 * Implementations in the `simple` namespace are very small and simple at the
 * expense of efficiency. Their purpose is to be easily identified as valid
 * implementations of the corresponding algorithms and to produce results for
 * small inputs that can be compared with results of other implementations.
 * 
 * Implementations in the `efficient` namespace have an asymptotically optimal
 * runtime and can be used to compute millions of digits of pi, but they return
 * exact rational numbers as results which implies a space complexity of 
 * O(n log n). Their results should be precisely equal to the ones in the
 * simple namespace.
 * 
 * Implementations in the `optimized` namespace are written for the express
 * purpose of computing results of the algorithms to a certain number of digits
 * of precision. They can be used to calculate digits of pi in O(n (log n)^3)
 * time and O(n) space (assuming O(n log n) time for multiplication).
 * 
 * None of the implementations utilize multithreading and there are likely many
 * more potential optimizations.
 *
 * There should be further information at the GitHub repository at
 * https://github.com/Shakatir/picalc
 */

#include <gmpxx.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <utility>

namespace picalc
{

namespace simple
{

/*
Calculates the result of the Dimension Specific Algorithm in dimension dim and
count iterations.

The implementation is very simple and inefficient. For reasonable execution
time, the function should not be called with arguments greater than a few
thousand.
*/
inline mpq_class dimension_specific(
	mp_limb_signed_t dim,
	mp_limb_signed_t count)
{
	assert(dim >= 1);
	assert(count >= 0);

	mpq_class sum = 0;
	mpq_class factor = 2;

	for (mp_limb_signed_t i = 1; i <= dim; ++i) {
		factor *= 2 * i;
		factor /= 2 * i - 1;
	}

	for (mp_limb_signed_t i = 0; i <= count; ++i) {
		sum += factor;

		factor *= (2 * i + 1);
		factor *= (2 * i + 1 - 2 * dim);
		factor /= (2 * i + 2);
		factor /= (2 * i + 3);
	}

	return sum;
}

/*
Calculates the result of the Trans Dimensional Algorithm after n iterations.

The implementation is very simple and inefficient. For reasonable execution
time, the function should not be called with arguments greater than a few
thousand.
*/
inline mpq_class trans_dimensional(
	mp_limb_signed_t n)
{
	assert(n >= 0);

	mpq_class sum = 2;

	for (mp_limb_signed_t i = 0; i < n; ++i) {
		mpq_class bi(2, 2 * i + 1);
		mpq_class ci(-2 * (i + 1), 2 * i + 1);
		ci /= (4 * i + 3);
		mpq_class di(-1, 2 * (4 * i + 5));
		mpq_class factor((i & 1) ? -1 : 1, mpz_class(1) << (2 * i));
		sum += factor * (bi + ci + di);
	}

	return sum;
}

} // namespace simple

namespace efficient
{

namespace detail
{

struct partial
{
	mpq_class sum;
	mpq_class factor;
};

inline partial combine(const partial& l, const partial& r)
{
	return {l.sum + l.factor * r.sum, l.factor * r.factor};
}

inline mpz_class gen_factorial(
	mp_limb_signed_t first,
	mp_limb_signed_t step,
	mp_limb_signed_t count)
{
	if (count <= 0) {
		return 1;
	} else if (count == 1) {
		return first;
	} else {
		return gen_factorial(first, step * 2, count - count / 2)
			   * gen_factorial(first + step, step * 2, count / 2);
	}
}

inline partial dim_spec_init(
	mp_limb_signed_t dim)
{
	mpq_class ret_factor(2 * gen_factorial(2, 2, dim), gen_factorial(1, 2, dim));
	ret_factor.canonicalize();
	return {0, std::move(ret_factor)};
}

inline partial dim_spec_it(
	mp_limb_signed_t dim,
	mp_limb_signed_t first,
	mp_limb_signed_t count)
{
	assert(dim >= 1);
	assert(first >= 0);
	assert(count >= 0);

	if (count <= 0) {
		return {0, 1};
	} else if (count == 1) {
		mpz_class num_factor = 2 * first + 1;
		num_factor *= 2 * first + 1 - 2 * dim;
		mpz_class den_factor = 2 * first + 2;
		den_factor *= 2 * first + 3;
		
		mpq_class ret_factor(std::move(num_factor), std::move(den_factor));
		ret_factor.canonicalize();
		return {1, std::move(ret_factor)};
	} else {
		return combine(dim_spec_it(dim, first, count / 2),
			dim_spec_it(dim, first + count / 2, count - count / 2));
	}
}

inline partial trans_dim_init()
{
	return {2, {37, 30}};
}

inline partial trans_dim_it(
	mp_limb_signed_t first,
	mp_limb_signed_t count)
{
	assert(first >= 0);
	assert(count >= 0);

	if (count <= 0) {
		return {0, 1};
	} else if (count == 1) {
		constexpr long num_coeffs[6]{-1280, -8384, -20528, -23364, -12288, -2385};
		constexpr long den_coeffs[6]{5120, 38656, 113344, 160592, 109056, 27972};

		mpz_class num = num_coeffs[0];
		mpz_class den = den_coeffs[0];
		for (int i = 1; i < 6; ++i) {
			num *= first;
			den *= first;
			num += num_coeffs[i];
			den += den_coeffs[i];
		}

		return {1, {num, den}};
	} else {
		return combine(trans_dim_it(first, count / 2),
			trans_dim_it(first + count / 2, count - count / 2));
	}
}

} // namespace detail

/*
Calculates the result of the Dimension Specific Algorithm in dimension dim and
count iterations.

The implementation is asymptotically optimal, but written as a compromise
between simplicity and efficiendy.
*/
inline mpq_class dimension_specific(
	mp_limb_signed_t dim,
	mp_limb_signed_t iter)
{
	return combine(detail::dim_spec_init(dim), detail::dim_spec_it(dim, 0, iter + 1)).sum;
}

/*
Calculates the result of the Trans Dimensional Algorithm after n iterations.

The implementation is asymptotically optimal, but written as a compromise
between simplicity and efficiendy.
*/
inline mpq_class trans_dimensional(
	mp_limb_signed_t n)
{
	return combine(detail::trans_dim_init(), detail::trans_dim_it(0, n)).sum;
}

} // namespace efficient

namespace optimized
{

namespace detail
{

struct partial
{
	mpz_class num_sum;
	mpz_class num_factor;
	mpz_class denom;
};

inline partial combine(const partial& l, const partial& r)
{
	return {
		l.num_sum * r.denom + l.num_factor * r.num_sum,
		l.num_factor * r.num_factor,
		l.denom * r.denom};
}

struct result
{
	mpz_class sum;
	mpz_class factor;

	mp_limb_signed_t base;
	mp_limb_signed_t digits;
	mp_limb_signed_t padding;
};

inline void apply(
	result& res,
	const partial& p)
{
	res.sum += res.factor * p.num_sum / p.denom;
	res.factor = res.factor * p.num_factor / p.denom;
}

using efficient::detail::gen_factorial;

inline result dim_spec_init(
	mp_limb_signed_t dim,
	mp_limb_signed_t base,
	mp_limb_signed_t digits)
{
	assert(dim >= 1);
	assert(base >= 2 && base <= 36);
	assert(digits >= 0);

	result ret{0, 0, base, digits, 17 * dim / 32 + 100};
	mpz_ui_pow_ui(ret.factor.get_mpz_t(), base, digits);
	ret.factor <<= ret.padding + dim + 1;

	mp_limb_signed_t slices = std::ceil((std::log2(dim) * dim) / (std::log2(base) * digits + ret.padding) / 2) + 1;

	for (mp_limb_signed_t i = 0; i < slices; ++i) {
		mp_limb_signed_t count = (dim / slices) + (i < dim % slices);
		ret.factor *= gen_factorial(1 + i, slices, count);
		ret.factor /= gen_factorial(2 * dim - 1 - 2 * i, -2 * slices, count);
	}

	return ret;
}

inline partial dim_spec_it(
	mp_limb_signed_t dim,
	mp_limb_signed_t first,
	mp_limb_signed_t count)
{
	assert(dim >= 1);
	assert(first >= 0);
	assert(count >= count);

	if (count <= 0) {
		return {0, 1, 1};
	} else if (count == 1) {
		partial ret{};
		ret.num_factor = (2 * first + 1);
		ret.num_factor *= (2 * first + 1 - 2 * dim);
		ret.denom = (2 * first + 2);
		ret.denom *= (2 * first + 3);
		ret.num_sum = ret.denom;
		return ret;
	} else {
		return combine(dim_spec_it(dim, first, count / 2), dim_spec_it(dim, first + count / 2, count - count / 2));
	}
}

inline result trans_dim_init(
	mp_limb_signed_t base,
	mp_limb_signed_t digits)
{
	assert(base >= 2 && base <= 36);
	assert(digits >= 0);

	result ret{0, 0, base, digits, 100};
	mpz_ui_pow_ui(ret.factor.get_mpz_t(), base, digits);
	ret.factor <<= ret.padding;
	ret.sum = ret.factor;
	ret.sum <<= 1;
	ret.factor *= 37;
	ret.factor /= 30;
	return ret;
}

inline partial trans_dim_it(
	mp_limb_signed_t first,
	mp_limb_signed_t count)
{
	assert(first >= 0);
	assert(count >= 0);

	if (count <= 0) {
		return {0, 1, 1};
	} else if (count == 1) {
		constexpr long num_coeffs[6]{-1280, -8384, -20528, -23364, -12288, -2385};
		constexpr long den_coeffs[6]{5120, 38656, 113344, 160592, 109056, 27972};

		partial ret{};

		ret.num_factor = num_coeffs[0];
		ret.denom = den_coeffs[0];
		for (int i = 1; i < 6; ++i) {
			ret.num_factor *= first;
			ret.denom *= first;
			ret.num_factor += num_coeffs[i];
			ret.denom += den_coeffs[i];
		}

		ret.num_sum = ret.denom;

		return ret;
	} else {
		return combine(trans_dim_it(first, count / 2), trans_dim_it(first + count / 2, count - count / 2));
	}
}

} // namespace detail

/*
Calculates the result of the Dimension Specific Algorithm in dimension dim and
count iterations multiplied by (base raised to the power of digits), rounded
towards zero.
*/
inline mpz_class dimension_specific(
	mp_limb_signed_t dim,
	mp_limb_signed_t count,
	mp_limb_signed_t base,
	mp_limb_signed_t digits)
{
	assert(dim >= 1);
	assert(count >= 0);
	assert(base >= 2 && base <= 36);
	assert(digits >= 0);

	++count;

	detail::result res = detail::dim_spec_init(dim, base, digits);
	mp_limb_signed_t slices = std::ceil(((2 * std::log2(std::max(dim, count)) + 2) * count) / (std::log2(base) * digits + 1));

	mp_limb_signed_t total = 0;
	for (mp_limb_signed_t i = 0; i < slices; ++i) {
		mp_limb_signed_t it = count / slices + (i < count % slices);
		apply(res, detail::dim_spec_it(dim, total, it));
		total += it;
	}

	res.sum >>= res.padding;
	return std::move(res.sum);
}

/*
Calculates the result of the Trans-Dimensional Algorithm in after count
iterations multiplied by (base raised to the power of digits), rounded
towards zero.
*/
inline mpz_class trans_dimensional(
	mp_limb_signed_t count,
	mp_limb_signed_t base,
	mp_limb_signed_t digits)
{
	assert(count >= 0);
	assert(base >= 2 && base <= 36);
	assert(digits >= 0);

	detail::result res = detail::trans_dim_init(base, digits);
	mp_limb_signed_t slices = std::ceil(((5 * std::log2(count + 1) + 13) * (count + 1)) / (std::log2(base) * (digits + 1)) + 100);

	mp_limb_signed_t total = 0;
	for (mp_limb_signed_t i = 0; i < slices; ++i) {
		mp_limb_signed_t it = count / slices + (i < count % slices);
		apply(res, detail::trans_dim_it(total, it));
		total += it;
	}

	res.sum >>= res.padding;
	return std::move(res.sum);
}

} // namespace optimized

} // namespace picalc
