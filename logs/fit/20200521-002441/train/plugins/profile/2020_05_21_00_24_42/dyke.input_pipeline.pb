$	qxADj�O@r���6>V@����&%�?!sL��_@$	����O@ '��O�@���1�?!��"`�'@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$sL��_@Su�l�1_@A��z�?Y]p���?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&����&%�?�q�Z|
�?Aҍ�����?Y��۞ ��?*	
ףp�&�@2S
Iterator::Model::ParallelMap�lXSY�?!����VH@)�lXSY�?1����VH@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap����K��?!���=��@@)�5��,�?14����1@:Preprocessing2n
7Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch��U���?!U:^��K*@)��U���?1U:^��K*@:Preprocessing2s
<Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map�$����?!k(^^�$@)� m�Yg�?1��&��@:Preprocessing2F
Iterator::Model�eN����?!1v��,K@)�ȯb��?1q�>ů@:Preprocessing2�
JIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�f���?!�*�a�@)�ՏM�#�?1�� ZY@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[8]::Concatenate��Z(��?!��y(�?)�K���ԫ?16͘�jY�?:Preprocessing2X
!Iterator::Model::ParallelMap::Zip
ǜg�K��?!�g�h��A@)"���ɩ�?10����G�?:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::ConcatenateNa�����?!SB�X��?)k���#G�?1���"��?:Preprocessing2�
QIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeƈD�eݏ?!t�T)��?)ƈD�eݏ?1t�T)��?:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate�B�=��?!�������?)��Dׅ�?1p�^(4��?:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat4��𽿑?! �.ё��?)؛����?1B�3
�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor<J%<��o?!y����?)<J%<��o?1y����?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor�TPQ�+m?!Tf��Gþ?)�TPQ�+m?1Tf��Gþ?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�4�ׂ�[?!��h�c�?)�4�ׂ�[?1��h�c�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensore����`[?!£�5�ެ?)e����`[?1£�5�ެ?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor$�@�X?!0�W.��?)$�@�X?10�W.��?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice�A�p�-N?!���Lӟ?)�A�p�-N?1���Lӟ?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*high2B98.2 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
$	�9��MO@�E?���U@�q�Z|
�?!Su�l�1_@	!       "	!       *	!       2$	����
��?1dA�Q�?��z�?!ҍ�����?:	!       B	!       J$	�/��s�?�+���?��۞ ��?!]p���?R	!       Z$	�/��s�?�+���?��۞ ��?!]p���?JCPU_ONLY