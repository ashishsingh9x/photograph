MA practical 3:
Working with Docker
1) create Docker Hub account (sign up)
2) login to https://labs.play-with-docker.com/
Click on start
3)add new instance
4)perform following:
Method1:
To pull and push images using docker
Command: to check docker version
docker –version
output:
Command: to pull readymade image
docker pull rocker/verse
output:
Command: to check images in docker
docker images
output:
Now Login to docker hub and create repository
Output:
Click on Create button
Now check repository created
Command: to login to your docker account
docker login –username=kbdocker11
password:
note: kbdocker11 is my docker ID . You will use your docker ID here. And enter your
password .
Output:
Command : to tag image
docker tag 8c3e4e2c3e kbdocker11/repo1:firsttry
note: here 8c3e4e2c3e this is image id which you can get from docker images
command.
Output:
Command: to push image to docker hub account
docker push kbdocker11/repo1:firsttry
note: firsttry is tag name created above.
Output
Check it in docker hub now
Click on tags and check
Method 2:
Build an image then push it to docker and run it
Command : to create docker file
1. cat > Dockerfile <<EOF
2. FROM busybox
3. CMD echo "Hello world! This is my first Docker image."
4. EOF
Output:
Command : to build image from docker file
 dokcer build –t kbdocker11/repo2 .
Output:
Command: to check docker images
 docker images
output:
Command: to push image to docker hub
 docker push kbdocker11/repo2 .
Output:
Now check it on docker hub
command: to run docker image:
 docker run kbdocker11/repo2
output:
Now close session. 
