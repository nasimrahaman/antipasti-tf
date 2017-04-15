# Antipasti-TF

Antipasti-TF is a lightweight wrapper around Tensorflow for building 
convolutional neural networks with complex architechtures. In under 
active development and not ready for deployment - yet. :-)

## Planned API

The API is non-functional and uses [NetworkX](https://networkx.github.io/) behind the scenes, 
which allows for arbitrarily complex architectures.  

For the following, let's assume that our model is called `network`.   

* We implement an API resembling python lists/tuples for sequentially 
  stacking layers: `network = conv(...) + pool(...) + conv(...) + ...`. 
  This supports constructs like list comprehensions and reductions. 
  In addition, we define a `*` operation to stack layers laterally, 
  so you could define an [inception module](http://wikicoursenote.com/wiki/File:I1.png) like: 
  `previous + conv(...) * conv(...) * conv(...) * pool(...) + concat() + next`. 
  For predefined sub-networks as modules, a [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) could look like: 
  `network = module_1 + (module_2 + module_3 * id() + module_4) * id() + module_5`.
  Slicing would be defined as expected, e.g. `subnetwork = network[3:5]` 
  and network surgery would be as simple as `network[5:8] = another_network[3:5]`. 

* Simultaneously, we expose as much of the graph as possible, so you could do 
  things like `network.add_nodes(['conv1', 'pool1', ..., 'conv10', ...])` 
  followed by `network.add_connection(from_layer='conv1', to_layer='conv10', join_by='concatenation')`, 
  or simply `network.add_layer(conv(...), previous_layer='pool8')`.


## Why not [Keras](https://github.com/fchollet/keras), a mature library with eloquent code?

If you're reading this, we assume you're somewhat familiar with Keras. 
Without further ado, there are three reasons:   
   
   * Keras is [functional](https://keras.io/getting-started/functional-api-guide/) for the most part. The functional API can be 
     powerful for constructing models, but altering a model after it 
     has been built is usually a tedious process. We're aiming for 
     a non-functional API, one that affords flexibility before and after 
     the model has been built. This enables efficient [net surgery](https://github.com/BVLC/caffe/blob/master/examples/net_surgery.ipynb)
     (as it's fondly called in the Caffe community) on pretrained models.
     Think of it as a keras Sequential model, except it doesn't need to 
     be sequential. 
     
   * Keras is a multi-framework tool, wrapping both Theano and Tensorflow. 
     Theano and Tensorflow are awesome libraries, but we have found 
     that supporting both simultaneously does neither justice while 
     slowing down the development process. With Antipasti, we're shooting 
     for a library that fully integrates with Tensorflow (including
     graceful and transparent handling of data- and model-parallel 
     multi-GPU training (synchronous and asynchronous) out of the box, 
     and full support for distributed tensorflow).
     
   * Keras is a general purpose framework, but Antipasti is being built 
     with images and volumes as first class citizens. A fully connected 
     layer is just a convolutional layer with a `1x1` filter.     
     
   * Antipasti does not intend to replace Keras. In fact, it should add 
     to it by implementing a multi-GPU/distributed training bench for 
     Keras models via a Keras to Antipasti compatibility layer.
     
## Getting Involved

Help shape this project! Have suggestions on how the API should look, or 
time to contribute? Get in touch by email (nasim.rahaman at iwr.uni-heidelberg.de) 
or opening an issue.         

## Who's involved?

As of the day: 

* Nasim Rahaman of [Image Analysis and Learning Lab](https://hci.iwr.uni-heidelberg.de/mip) 
@ [Heidelberg Collaboratory for Image Processing](https://hci.iwr.uni-heidelberg.de/).