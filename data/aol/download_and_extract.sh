#!/usr/bin/env bash
wget -O session_train.zip "https://drive.google.com/uc?export=download&id=0B8ZGlkqDw7hFUzViMXE4akp4NGM"
wget -O session_dev.zip "https://drive.google.com/uc?export=download&id=0B8ZGlkqDw7hFZkNsb3hLSnVoamc"
wget -O session_test.zip "https://drive.google.com/uc?export=download&id=0B8ZGlkqDw7hFUlFSRzVHaWhCWGs"
unzip session_train.zip
unzip session_dev.zip
unzip session_test.zip
rm session_train.zip
rm session_dev.zip
rm session_test.zip
