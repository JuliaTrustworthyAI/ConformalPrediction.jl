# Following https://ahsmart.com/pub/holy-traits-design-patterns-and-best-practice-book/

# SAMPLING STYLE

# Defining the trait type
abstract type SamplingType end
struct IsInductive <: SamplingType end
struct IsTransductive <: SamplingType end

# Identifying traits
SamplingType(::Type) = IsInductive()                            # default behaviour is inductive
SamplingType(::Type{<:TransductiveModel}) = IsTransductive()

# Implementating trait behaviour
requires_data_splitting(x::T) where {T} = requires_data_splitting(SamplingType(T), x)
requires_data_splitting(::IsInductive, x) = true
requires_data_splitting(::IsTransductive, x) = false
