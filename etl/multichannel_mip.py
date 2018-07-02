"""
Purpose: This script implements maximum intensity projections (MIP). This
process involves taking 3D brain scans, chunking them into three relevant
sections, and compressing each section's maximum values down into a 2D array.
When these are recombined, we get an array with the shape (3, X, Y) — which is
ready to be fed directly into standardized Keras architecture with pretrained
feature detection weights from ImageNet/CIFAR10.
"""

import logging
# from matplotlib import pyplot as plt
import lib.cloud_management as cloud
import lib.transforms as transforms

WHENCE = ['numpy/axial',
          'numpy/coronal']

FAILURE_ANALYSIS = ['SSQSB70QYC5L5CJJ',
                    'MTWW42SPCGLHEDKY',
                    'GKDV3FW4M56I3IKV',
                    'BMFSUHVBPJ7RY56P',
                    'NWO33W453F6ZJ4BU',
                    '8TLM2DUBYEE2GDH0',
                    'J3JHPC3E15NZGX34',
                    'J3JHPC3E15NZGX35',
                    '3AWM4ZZHCWJ8MREY',
                    'MHL54TPCTOQJ2I4K',
                    '5H94IH9XGI83T610',
                    'UGXVSPJLHJL6AHSW',
                    '2KMKXR2G1BLD0C2G',
                    'PCNMFAZL5VWWK7RP',
                    'VVGO45TQNOASBLZM',
                    'NHXCOHZ4HH53NLQ6',
                    'F8W2RDY3D6L2EOFT',
                    'KK2Y9XHUUUC5LISA',
                    'HZLBHRHYLSY9TXJ4',
                    '0RB9KGMO90G1YQZD',
                    'J2JPFK8ZOICHFG34',
                    'HLXOSVDF27JWNCMJ',
                    'AEPRN5R7W2ASOGR0',
                    '99YJX0CY4FHHW46S',
                    'LOZOKQFJMDLL6IL5',
                    'STCSWQHX4UN23CDK',
                    'ZC9H37RWIQ90483S',
                    'CJDXGZHXAGH7QL3C',
                    '5KZSOKNYS84ZTDK6',
                    'AKZ8T688ZRU9UTY2',
                    '6BMRRS9RAZUPR3IL',
                    'HXLMZWH3SFX3SPAN',
                    'GHVG2CNNRZ65UBEU',
                    'TSDXCC6X3M7PG91E',
                    'IP4X9W512RO56NQ7',
                    'LNU3P20QOML7YGMZ',
                    '56GX2GI8AGT0BIHN',
                    'TSZFE43KG3NQJR69',
                    'IXRSXXZI0S6L0EJI',
                    'FCYGZ75WMW6L4PJM',
                    'KOE9CU24WK2TUQ43',
                    'KOE9CU24WK2TUQ44',
                    'TFIG39JA77W6USD3',
                    'XQBRGW3CYGNUMWHI',
                    'PB7YJZRJU74HFKTS',
                    'EUNTRXNEDB7VDVIS',
                    'JQWIIAADGKE2YMJS',
                    'NXLFQLVZRLUEK2UF',
                    'RHWTMTHC7HIWZ2YZ',
                    'RW13S0OR03CO7OP5',
                    'LYUO2OPTNYUBCHT2',
                    'CSCIKXOMNAIB3LUQ',
                    'Q8BNE59JIKQLLYJ1',
                    'K2GS9PIQ1E0DBDBE',
                    'WWEFFBIMLZ3KLQVZ']


def configure_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def multichannel_mip():
    configure_logger()
    client = cloud.authenticate()
    bucket = client.get_bucket('elvos')

    # iterate through every source directory...
    for location in WHENCE:
        prefix = location + '/'
        logging.info(f"MIPing images from {prefix}")

        for in_blob in bucket.list_blobs(prefix=prefix):
            # blacklist
            if in_blob.name == prefix + 'LAUIHISOEZIM5ILF.npy':
                continue

            file_id = in_blob.name.split('/')[2]
            file_id = file_id.split('.')[0]

            # perform the normal MIPing procedure
            logging.info(f'downloading {in_blob.name}')
            input_arr = cloud.download_array(in_blob)
            logging.info(f"blob shape: {input_arr.shape}")
            if file_id in FAILURE_ANALYSIS:
                if location == 'numpy/axial':
                    cropped_arr = \
                        transforms.crop_multichannel_axial_fa(input_arr,
                                                              location)
            else:
                if location == 'numpy/axial':
                    cropped_arr = transforms.crop_normal_axial(input_arr,
                                                               location)
                else:
                    cropped_arr = transforms.crop_normal_coronal(input_arr,
                                                                 location)
            not_extreme_arr = transforms.remove_extremes(cropped_arr)
            logging.info(f'removed array extremes')
            mip_arr = transforms.mip_multichannel(not_extreme_arr)
            # plt.figure(figsize=(6, 6))
            # plt.imshow(mip_arr[1], interpolation='none')
            # plt.show()

            # if the source directory is one of the luke ones
            # if location != 'numpy':
            #     file_id = in_blob.name.split('/')[2]
            #     file_id = file_id.split('.')[0]
            #     # save to both a training and validation split
            #     # and a potential generator source directory
            #     cloud.save_npy_to_cloud(mip_arr, file_id, 'processed')
            # # otherwise it's from numpy
            # else:
            # save to the numpy generator source directory
            cloud.save_npy_to_cloud(mip_arr, file_id, location, 'multichannel')


if __name__ == '__main__':
    multichannel_mip()
