## Guides

**jupyter notebook:** Video guide https://drive.google.com/file/d/1XUYxPthMsnSL37nbxGq30uDIo4mdnbn7/view?usp=sharing

- No password, use external IP address of the instance to connect on the port 9999 (eg: http://IP_address:9999)


## Useful commands

**extract with progress bar (usefull for large datasets, without printing all file names)**
```pv nsynth-train.jsonwav.tar.gz | tar -xz```

**setup jupyter for remote access if not setup already (for new env):**
https://ecbme4040.github.io/2022_fall/EnvSetup/gcp.html

**Run in background with screen commands**
Video guide https://drive.google.com/file/d/1SS823dc7DCoNTXItxpVs0U4xB_eE0cTc/view?usp=sharing
- create new screen: ``` screen ```
- list screens: ```screen -ls```
- detach screen:``` screen -d SCREEEN_ID```
- attach screen:```screen -r SCREEN_ID```

## GCP Buckets cheat sheet

https://cloud.google.com/storage/docs/uploading-objects

**upload file to a bucket:** ```gcloud storage cp OBJECT_LOCATION gs://DESTINATION_BUCKET_NAME/FILEPATH.EXT```

**upload folder:** ```gcloud storage cp -r OBJECT_LOCATION gs://DESTINATION_BUCKET_NAME/FOLDERPATH```

**create folder:** in ui, under the cloud storage tab -> butcket -> capstone_datasets

**download from url directly to a bucket (compressed files cannot be unzipped within a bucket, they need to be downloaded to a vm, then uploaded again)**: ```curl URL | gsutil cp - gs://YOUR_BUCKET_NAME/FILENAME.EXTENSION```
**eg:** ```curl https://www.openslr.org/resources/12/test-clean.tar.gz | gsutil cp - gs://capstone_datasets/librispeech/test/test.tar.gz```

## Drivers

- First reset the VM the first time it boots.
- Run  ```sudo dpkg --configure -a ```

 - Download script ```curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py ```

 - Run script ```sudo python3 install_gpu_driver.py ```
 
 (source: https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)

