

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'Mcs_sepConv_3x3',
    'Mcs_sepConv_5x5',
    'Mcs_sepConv_7x7',
]
#---houston
HSI=Genotype(normal=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('Mcs_sepConv_3x3', 0), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('Mcs_sepConv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('Mcs_sepConv_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 3), ('Mcs_sepConv_5x5', 1), ('max_pool_3x3', 2)], reduce_concat=range(2, 6))
