JAX Notes

# Key concepts 
(https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html)

* Numpy inspired interface

* functional programming paradigm -> necessity of pure functions without side effects (e.g no hidden class states)

* Transform numerical functinos via composable transformations (taken from Deep Learning with JAX, Ch. 1.2.2)
	* Taking derivatives with jax.grad (autodiff)
	* Compiling code with Just in Time (JIT) compilation
		* compiles and produces efficient code for GPUs and TPUs (even CPU in some cases)
	* Auto-vectorizing with jax.vmap()
		* takes care of batch dimensions of arrays and converts code from processing single item of data to batch (efficiently parallelize matrix computations)
	* Parallelizing code to run on multiple accelerators with jax.pmap()




# The Sharp Bits 
(https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)

* Pure functions: JAX transformatinos and compilation only work on functions whos output solely rely on the inputs (no hidden dynamic class variables!)
	-> "A function can still be pure when using stateful objects internally, as long as it does not read or write external state"

* Using jax.jit with class methods:
	* 1. easiest method is to create JIT-decorated helper function external to the class
	* 2. making self static, BUT care has to be taken and appropriate __hash__ and __eq__ methods have to be defined in order to get different results from multiple function calls
	* 3. making class a PyTree, requires flatten and unflatten functions

* Unlike np arrays jax arrays are always immutable 
	-> contents cant change once created
	-> JAX provides set function to do updates (e.g. .at[index].set())

* Out-of-bounds indexing: usually last element of array is returned instead of error or NaN
	-> use .at[index].get()

* Non-array inputs: jnp functions generally only accept arrays




# Pseudorandom Numbers 
(https://docs.jax.dev/en/latest/random-numbers.html#pseudorandom-numbers)

* Numpy uses "Mersenne Twister PRNG", which uses a global state under the hood
* -> this makes it hard to be 1. reproducible while 2. being parallelizable and 3. vectorisable due to dependencies on the global state
* to achieve these properties JAX sticks to its general programming paradigm and tracks PRNG state explicitely via a random key (which is effectively a stand in for numpys hidden state object)
* this key gets passed explicitely to random functions
* same key always leads to same output -> rule of thumb: never reuse keys to avoid unwanted correlations
* use split() function to generate new independant keys
* jax does not provide sequential equivalence guarantee (e.g. when generating a random vector constructed by a key split in 3 the result will differ from using the keys all at once using vmap or not splitting the key and specifying the vector shape in the distribution function)



# Extension libraries

* Optax: gradient processing and optimization library for JAX
	* https://github.com/google-deepmind/optax

* Optimistix: nonlinear optimization 
	* newer and smaller userbase compared to Optax
	* https://github.com/patrick-kidger/optimistix

* Diffrax: numerical differential equation solver 
	* https://docs.kidger.site/diffrax/

* Awesome JAX: curated list of JAX libraries
	* https://github.com/n2cholas/awesome-jax