# # MIT License
# # 
# # Copyright (c) 2024 Zihan Zhang, Yi Zhao, Harbin Institute of Technology
# # 
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# # 
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# # 
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.


# from transformers import PreTrainedTokenizer
# from scipy import io
# import pickle
# import random
# import numpy
# import mat73
# import torch
# import h5py
# import math
# import os
# import re


# def get_imagine(sub: str, epoch: int):
#     paths = f"./Chisco/derivatives/preprocessed_pkl/sub-{sub}/sub-{sub}_task-imagine_run-0{str(epoch)}_eeg.pkl"
#     if not os.path.exists(paths): return list()
#     pickles = pickle.load(open(paths, "rb"))
#     print(sub, epoch, len(pickles))

#     for idx, trial in enumerate(pickles):
#         assert isinstance(trial['input_features'], numpy.ndarray)
#         assert trial['input_features'].dtype == numpy.float64
#         assert trial['input_features'].shape == (1, 125, 1651)
#         input_features = trial['input_features'][0, :122, :]*1000000
#         mean = numpy.absolute(numpy.mean(input_features, axis=1))
#         stds = numpy.std(input_features, axis=1)
#         assert isinstance(input_features, numpy.ndarray)
#         assert input_features.dtype == numpy.float64
#         assert input_features.shape == (122, 1651)
#         assert (mean > 0).all() and (mean < 10000).all()
#         assert (stds > 0).all() and (stds < 10000).all()
#     return pickles

# def get_dataset(sub: str):
#     dsplit = {"input_features": [], "labels": []}
#     for epoch in range(1, 46):
#         pickles = get_imagine(sub=sub, epoch=epoch)
#         for trial in pickles:
#             input_features = trial['input_features'][0, :122, :]*1000000
#             input_ids = trial['text'].strip()

#             input_features = numpy.float32(input_features)
#             input_features = torch.tensor(input_features)
#             dsplit["input_features"].append(input_features)
#             dsplit["labels"].append(input_ids)
#     return dsplit




# MIT License
# 
# Copyright (c) 2024 Zihan Zhang, Yi Zhao, Harbin Institute of Technology
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from transformers import PreTrainedTokenizer
from scipy import io
import pickle
import random
import numpy
import mat73
import torch
import h5py
import math
import os
import re


def get_imagine(sub: str, epoch: int):
    paths = f"./Chisco/derivatives/preprocessed_pkl/sub-{sub}/sub-{sub}_task-imagine_run-0{str(epoch)}_eeg.pkl"
    print(f"Loading {paths} ...")
    if not os.path.exists(paths): return list()
    pickles = pickle.load(open(paths, "rb"))
    print(sub, epoch, len(pickles))

    for idx, trial in enumerate(pickles):
        assert isinstance(trial['input_features'], numpy.ndarray)
        assert trial['input_features'].dtype == numpy.float64
        assert trial['input_features'].shape == (1, 125, 1651)
        input_features = trial['input_features'][0, :122, :]*1000000
        mean = numpy.absolute(numpy.mean(input_features, axis=1))
        stds = numpy.std(input_features, axis=1)
        assert isinstance(input_features, numpy.ndarray)
        assert input_features.dtype == numpy.float64
        assert input_features.shape == (122, 1651)
        assert (mean > 0).all() and (mean < 10000).all()
        assert (stds > 0).all() and (stds < 10000).all()
    return pickles

def get_dataset(sub: str):
    dsplit = {"input_features": [], "labels": []}
    for epoch in range(1, 46):
        pickles = get_imagine(sub=sub, epoch=epoch)
        for trial in pickles:
            input_features = trial['input_features'][0, :122, :]*1000000
            input_ids = trial['text'].strip()

            input_features = numpy.float32(input_features)
            input_features = torch.tensor(input_features)
            dsplit["input_features"].append(input_features)
            dsplit["labels"].append(input_ids)
    return dsplit

# def get_dataset(sub: str):
#     """
#     sub: can be a single subject id (e.g. "01"), a comma-separated list (e.g. "01,02,03"), or "all"
#     """
#     dsplit = {"input_features": [], "labels": []}
#     # Find all subject folders if sub == "all"
#     if sub == "all":
#         sub_dirs = [d for d in os.listdir("./Chisco/derivatives/preprocessed_pkl") if d.startswith("sub-")]
#         sub_list = [d.replace("sub-", "") for d in sub_dirs]
#     else:
#         sub_list = [s.strip() for s in sub.split(",")]

#     for sub_id in sub_list:
#         for epoch in range(1, 46):
#             pickles = get_imagine(sub=sub_id, epoch=epoch)
#             for trial in pickles:
#                 input_features = trial['input_features'][0, :122, :]*1000000
#                 input_ids = trial['text'].strip()
#                 input_features = numpy.float32(input_features)
#                 input_features = torch.tensor(input_features)
#                 dsplit["input_features"].append(input_features)
#                 dsplit["labels"].append(input_ids)
#     return dsplit
