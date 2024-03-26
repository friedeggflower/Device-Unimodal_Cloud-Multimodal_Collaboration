# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD= /Public_Dataset/Cloud_multimodal/checkpoint

if [ ! -d $DOWNLOAD/pretrained ] ; then
    mkdir -p $DOWNLOAD/pretrained
fi

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'
wget $BLOB/pretrained/uniter-base.pt -O $DOWNLOAD/pretrained/uniter-base.pt

