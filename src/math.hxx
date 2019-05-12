#pragma once

#ifdef CUDA_VERSION
    #define GLM_FORCE_CUDA
#endif


//#define GLM_FORCE_CXX17
//#define GLM_ENABLE_EXPERIMENTAL
//#define GLM_FORCE_RADIANS
//#define GLM_FORCE_DEPTH_ZERO_TO_ONE
//#define GLM_GTX_intersect
//
//#include <glm/glm.hpp>
//#include <glm/gtx/intersect.hpp>
//#include <glm/gtx/matrix_decompose.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/matrix_inverse.hpp>
//#include <glm/gtc/type_ptr.hpp>


#ifdef CUDA_VERSION
    #define CUDA_HOST __host__
    #define CUDA_DEVICE __device__
    #define CUDA_HOST_DEVICE __host__ __device__
#else
    #define CUDA_HOST
    #define CUDA_DEVICE
    #define CUDA_HOST_DEVICE
#endif


namespace math {
    template<class T>
    constexpr T radians(T degrees)
    {
        static_assert(std::numeric_limits<T>::is_iec559, "'radians' only accept floating-point input");

        return degrees * static_cast<T>(0.01745329251994329576923690768489);
    }

    template<class T>
    constexpr T degrees(T radians)
    {
        static_assert(std::numeric_limits<T>::is_iec559, "'degrees' only accept floating-point input");

        return radians * static_cast<T>(57.295779513082320876798154814105);
    }

    template<std::size_t D, class T>
    struct vec final {
        using value_type = T;
        static auto constexpr dimension{D};
    };

    template<class T>
    struct vec<3, T> final {
        using type = vec<3, T>;

        T x, y, z;

        vec() noexcept = default;
        ~vec() = default;

        CUDA_HOST_DEVICE constexpr vec(type const &vector) noexcept
        {
            this->x = vector.x;
            this->y = vector.y;
            this->z = vector.z;
        }

        CUDA_HOST_DEVICE constexpr vec(type &&vector) noexcept
        {
            this->x = vector.x;
            this->y = vector.y;
            this->z = vector.z;
        }

        CUDA_HOST_DEVICE constexpr type &operator= (type const &vector) noexcept
        {
            this->x = vector.x;
            this->y = vector.y;
            this->z = vector.z;

            return *this;
        }

        CUDA_HOST_DEVICE constexpr type &operator= (type &&vector) noexcept
        {
            this->x = vector.x;
            this->y = vector.y;
            this->z = vector.z;

            return *this;
        }

        template<class X, class Y, class Z>
        CUDA_HOST_DEVICE constexpr vec(X x, Y y, Z z) noexcept
            : x{static_cast<T>(x)}, y{static_cast<T>(y)}, z{static_cast<T>(z)} { }
        
        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE constexpr vec(S value) noexcept
            : x{static_cast<T>(value)}, y{static_cast<T>(value)}, z{static_cast<T>(value)} { }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type operator+ (V &&rhs) const
        {
            return {x + rhs.x, y + rhs.y, z + rhs.z};
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type operator- (V &&rhs) const
        {
            return {x - rhs.x, y - rhs.y, z - rhs.z};
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type operator* (V &&rhs) const
        {
            return {x * rhs.x, y * rhs.y, z * rhs.z};
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type operator/ (V &&rhs) const
        {
            return {x / rhs.x, y / rhs.y, z / rhs.z};
        }

        CUDA_HOST_DEVICE type &operator+ () noexcept { return *this; };
        CUDA_HOST_DEVICE type operator- () const noexcept { return {-x, -y, -z}; };

        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE type operator+ (S scalar) const
        {
            return {x + scalar, y + scalar, z + scalar};
        }
        
        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE type operator- (S scalar) const
        {
            return {x - scalar, y - scalar, z + scalar};
        }

        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE type operator* (S scalar) const
        {
            return {x * scalar, y * scalar, z * scalar};
        }

        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE type operator/ (S scalar) const
        {
            return {x / scalar, y / scalar, z / scalar};
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type &operator+= (V &&rhs)
        {
            this->x += rhs.x;
            this->y += rhs.y;
            this->z += rhs.z;

            return *this;
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type &operator-= (V &&rhs)
        {
            this->x -= rhs.x;
            this->y -= rhs.y;
            this->z -= rhs.z;

            return *this;
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type &operator*= (V &&rhs)
        {
            this->x *= rhs.x;
            this->y *= rhs.y;
            this->z *= rhs.z;

            return *this;
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type &operator/= (V &&rhs)
        {
            this->x /= rhs.x;
            this->y /= rhs.y;
            this->z /= rhs.z;

            return *this;
        }

        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE type &operator+= (S scalar)
        {
            this->x += scalar;
            this->y += scalar;
            this->z += scalar;

            return *this;
        }

        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE type &operator-= (S scalar)
        {
            this->x -= scalar;
            this->y -= scalar;
            this->z -= scalar;

            return *this;
        }

        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE type &operator*= (S scalar)
        {
            this->x *= scalar;
            this->y *= scalar;
            this->z *= scalar;

            return *this;
        }

        template<class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
        CUDA_HOST_DEVICE type &operator/= (S scalar)
        {
            this->x /= scalar;
            this->y /= scalar;
            this->z /= scalar;

            return *this;
        }

        CUDA_HOST_DEVICE float norm() const
        {
            return x * x + y * y + z * z;
        }

        CUDA_HOST_DEVICE float length() const
        {
            return std::sqrt(norm());
        }

        CUDA_HOST_DEVICE type &normalize()
        {
            auto length = this->length();

            if (std::abs(length) > /*std::numeric_limits<T>::min()*/static_cast<T>(1.e-6))
                *this /= length;

            return *this;
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE float dot(V &&rhs) const
        {
            return x * rhs.x + y * rhs.y + z * rhs.z;
        }

        template<class V, typename std::enable_if_t<std::is_same_v<std::decay_t<V>, type>>* = 0>
        CUDA_HOST_DEVICE type cross(V &&rhs) const
        {
            return {
                y * rhs.z - z * rhs.y,
                z * rhs.x - x * rhs.z,
                x * rhs.y - y * rhs.x
            };
        }
    };

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE float length(vec<D, T> const &vec)
    {
        return vec.length();
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE float length(vec<D, T> &&vec)
    {
        return vec.length();
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE float distance(vec<D, T> const &lhs, vec<D, T> const &rhs)
    {
        return length(lhs - rhs);
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE vec<D, T> normalize(vec<D, T> const &vec)
    {
        auto copy{vec};

        return copy.normalize();
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE vec<D, T> normalize(vec<D, T> &&vec)
    {
        return vec.normalize();
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE float dot(vec<D, T> const &lhs, vec<D, T> const &rhs)
    {
        return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE vec<D, T> cross(vec<D, T> const &lhs, vec<D, T> const &rhs)
    {
        return {
            lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.z * rhs.x - lhs.x * rhs.z,
            lhs.x * rhs.y - lhs.y * rhs.x
        };
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE vec<D, T> reflect(vec<D, T> const &I, vec<D, T> const &N)
    {
        return I - N * dot(N, I) * static_cast<T>(2);
    }

    template<std::size_t D, class T, class A>
    CUDA_HOST_DEVICE vec<D, T> refract(vec<D, T> const &I, vec<D, T> const &N, A eta)
    {
        static_assert(std::numeric_limits<A>::is_iec559, "'refract' accepts only floating-point inputs");

        auto const dotValue = dot(N, I);
        auto const k = static_cast<T>(1) - eta * eta * (static_cast<T>(1) - dotValue * dotValue);

        return (I * eta - (N * std::sqrt(k) + dotValue * eta)) * static_cast<T>(k >= static_cast<T>(0));
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE vec<D, T> pow(vec<D, T> const &base, vec<D, T> const &exponent)
    {
        static_assert(D == 3, "uninplemented case");

        auto copy = base;

        copy.x = std::pow(copy.x, exponent.x);
        copy.y = std::pow(copy.y, exponent.y);
        copy.z = std::pow(copy.z, exponent.z);

        return copy;
    }

    template<std::size_t D, class T, class S, typename std::enable_if_t<std::is_arithmetic_v<std::decay_t<S>>>* = 0>
    CUDA_HOST_DEVICE vec<D, T> mix(vec<D, T> const &x, vec<D, T> const &y, S a)
    {
        return x * static_cast<T>(static_cast<S>(1) - a) + y * static_cast<T>(a);
    }

    template<std::size_t D, class T>
    CUDA_HOST_DEVICE vec<D, T> mix(vec<D, T> const &x, vec<D, T> const &y, vec<D, T> const &a)
    {
        return x * (vec<D, T>{1} - a) + y * a;
    }

    using vec3 = vec<3, float>;
    using u8vec3 = vec<3, std::uint8_t>;


    struct ray final {
        math::vec3 origin;
        math::vec3 direction;

        ray() = default;

        template<class T1, class T2>
        CUDA_DEVICE ray(T1 &&origin, T2 &&direction) noexcept : origin{std::forward<T1>(origin)}, direction{std::forward<T2>(direction)} { }

        CUDA_DEVICE math::vec3 unit_direction() const noexcept { return math::normalize(direction); }

        CUDA_DEVICE math::vec3 point_at(float t) const noexcept { return origin + direction * t; }
    };
}
