$	���3�c@[.I@G�k@V,~SX)�?!�����s@$	~Ow��� @G���K'@�D�)3��?!�($n��0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�����s@%Z�xZ�s@A��֦���?Y���?Q��?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V,~SX)�?���S�?A�������?Y�26t�?�?*	�|?5^�@2S
Iterator::Model::ParallelMap<�b��*�?!���`�FC@)<�b��*�?1���`�FC@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�Hh˹�?!D�#�B@)�p����?1>.�B4@:Preprocessing2n
7Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch9�� n�?!����`&@)9�� n�?1����`&@:Preprocessing2s
<Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map�u�|�H�?!��=��.@)������?17DWJ�$@:Preprocessing2F
Iterator::Model�l��?!n��5�F@)������?1�3ؓ�s@:Preprocessing2�
JIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat)�k{�%�?!N́�1@)��'c|��?1GyVg@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[8]::Concatenate\����o�?!��u	�@)���oaݠ?1Pe���@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip
���=j�?!��
�C@)L3�뤾�?1�̴��?:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�}8H��?!�9P�I�?)�9�����?1ZT��*��?:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate���H.�?!�O"�ъ�?)����҈�?1���p��?:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�m��?!�++����?)�(�QGǅ?1���L��?:Preprocessing2�
QIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range5ӽN�˂?!W����?)5ӽN�˂?1W����?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor)��qh?!�l���?))��qh?1�l���?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor%Ί��>_?!_��p��?)%Ί��>_?1_��p��?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensor�_>Y1\]?!���'�?)�_>Y1\]?1���'�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�t�_��T?!��E���?)�t�_��T?1��E���?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensorj>"�DR?!p��-b�?)j>"�DR?1p��-b�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice;�f��O?!^�_�ư?);�f��O?1^�_�ư?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B99.6 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�]gC��c@��ea��k@���S�?!%Z�xZ�s@	!       "	!       *	!       2$	8K�r�?7DK�7(�?��֦���?!�������?:	!       B	!       J$	�z�G��? 䀷Ĝ�?�26t�?�?!���?Q��?R	!       Z$	�z�G��? 䀷Ĝ�?�26t�?�?!���?Q��?JCPU_ONLY