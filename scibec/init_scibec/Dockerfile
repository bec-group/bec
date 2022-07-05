# set base image (host OS)
FROM python:3.8-alpine

# set the working directory in the container
WORKDIR /code

# copy the content of the local opaas directory to the working directory
COPY ./scibec/init_scibec ./bec/scibec/init_scibec

WORKDIR /code/bec/scibec/init_scibec
RUN pip install requests pyyaml

# command to run on container start
CMD [ "python", "/code/bec/scibec/init_scibec/update_sessions.py", "--config", "/code/bec/scibec/init_scibec/test_config.yaml"]