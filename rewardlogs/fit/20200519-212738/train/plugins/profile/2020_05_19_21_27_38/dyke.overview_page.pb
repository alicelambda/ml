�$$	o~�D�t�?!��׀�?�f��j+�?!wI�Q��?$	��t�%57@Ej)�?/@ ���r(@!�ɅA@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$wI�Q��?I�2���?A��kЗ��?Y6��X�?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&�f��j+�?��`"�?A��(�[Z�?Y�l˟�?*	7�A`�z�@2S
Iterator::Model::ParallelMap�k�6�?!��.7�HB@)�k�6�?1��.7�HB@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap˟o��?!
�>�aA@)�Un2��?1����2@:Preprocessing2n
7Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch~��7L�?!��Y�W)@)~��7L�?1��Y�W)@:Preprocessing2s
<Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map)?����?!%ɭ�f93@)�2����?1~����(@:Preprocessing2�
JIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�>Ȳ`�?!��_-�%@)&�(�̵?1�ĳ�8�@:Preprocessing2F
Iterator::Model�+f���?!n����EE@)�|A�?1��7U�@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[8]::Concatenateu=�u��?!�+1�5-�?)Xr�ߔ?1�ҧim��?:Preprocessing2X
!Iterator::Model::ParallelMap::Zip
��6�4D�?!�H�� }B@)�7i͓?1� y��?:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�+����?!�@
�"��?)��$W@�?1[�a���?:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�r0� Ò?!9Έ�S�?)���g��?1��U�?:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate�~�n؆?!;U�G��?)t|�8c��?1;� ��E�?:Preprocessing2�
QIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range��PN���?!*(`=���?)��PN���?1*(`=���?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��h:;l?!`�a����?)��h:;l?1`�a����?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��`�$�_?!�J�&��?)��`�$�_?1�J�&��?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor�&���KZ?!Vw��_�?)�&���KZ?1Vw��_�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensor�PۆQP?!���$��?)�PۆQP?1���$��?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���3.L?!��F��?)���3.L?1��F��?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice��֪]C?!��l��?)��֪]C?1��l��?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[8]::Concatenate[0]::TensorSlice�_��s@?!�,��G��?)�_��s@?1�,��G��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 29.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2B44.6 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	d�3�%�?����$�?��`"�?!I�2���?	!       "	!       *	!       2$	XU/��d�?��)���?��kЗ��?!��(�[Z�?:	!       B	!       J$	֪]R�?�̮Ȳ%�?�l˟�?!6��X�?R	!       Z$	֪]R�?�̮Ȳ%�?�l˟�?!6��X�?JCPU_ONLY2black"�
host�Your program is HIGHLY input-bound because 29.1% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendationN
nohigh"B44.6 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 