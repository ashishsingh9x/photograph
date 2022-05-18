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

                          
 8) Running Location Service in Docker
(create docker hub login first to use it in play with docker)
Now login in to Play-With-Docker
Click on Start
Click on Add New Instance
Start typing following commands
Command : To run teamservice
docker run -d -p 5000:5000 -e PORT=5000 \
-e LOCATION__URL=http://localhost:5001 \
dotnetcoreservices/teamservice:location
output: (you can observe that it has started port 5000 on top)
Command: to run location service
docker run -d -p 5001:5001 -e PORT=5001 \
dotnetcoreservices/locationservice:nodb
output: (now it has started one more port that is 5001 for location service)
Command : to check running images in docker
docker images
output:
Command: to create new team
curl -H "Content-Type:application/json" -X POST -d \
'{"id":"e52baa63-d511-417e-9e54-7aab04286281", "name":"KC"}' http://localhost:5000/teams
Output:
Command :To confirm that team is added
curl http://localhost:5000/teams/e52baa63-d511-417e-9e54-7aab04286281
Output
Command : to add new member to team
curl -H "Content-Type:application/json" -X POST -d \
'{"id":"63e7acf8-8fae-42ce-9349-3c8593ac8292", "firstName":"Kirti", "lastName":"Bhatt"}'
http://localhost:5000/teams/e52baa63-d511-417e-9e54-7aab04286281/members
Output:
Command :To confirm member added
curl http://localhost:5000/teams/e52baa63-d511-417e-9e54-7aab04286281
output:
Command : To add location for member
curl -H "Content-Type:application/json" -X POST -d \
'{"id":"64c3e69f-1580-4b2f-a9ff-2c5f3b8f0e1f", "latitude":12.0,"longitude":12.0,"altitude":10.0,
"timestamp":0,"memberId":"63e7acf8-8fae-42ce-9349-3c8593ac8292"}' http://localhost:5001/locations/63e7acf8-
8fae-42ce-9349-3c8593ac8292
Output:
Command : To confirm location is added in member
curl http://localhost:5001/locations/63e7acf8-8fae-42ce-9349-3c8593ac8292
output: 
                          
6)Practical 6 (Working with Circle CI for continuous integration)
Step 1 - Create a repository
1. Log in to GitHub and begin the process to create a new repository.
2. Enter a name for your repository (for example, hello-world).
3. Select the option to initialize the repository with a README file.
4. Finally, click Create repository.
5. There is no need to add any source code for now.
Login to Circle CI https://app.circleci.com/ Using GitHub Login, Once logged in navigate to Projects.
Step 2 - Set up CircleCI
1. Navigate to the CircleCI Projects page. If you created your new repository under an organization, you will need to
select the organization name.
2. You will be taken to the Projects dashboard. On the dashboard, select the project you want to set up (hello-world).
3. Select the option to commit a starter CI pipeline to a new branch, and click Set Up Project. This will create a file
.circleci/config.yml at the root of your repository on a new branch called circleci-project-setup.
Step 3 - Your first pipeline
On your project’s pipeline page, click the green Success button, which brings you to the workflow that ran (say-helloworkflow).
Within this workflow, the pipeline ran one job, called say-hello. Click say-hello to see the steps in this job:
a. Spin up environment
b. Preparing environment variables
c. Checkout code
d. Say hello
Now select the “say-hello-workflow” to the right of Success status column
Select “say-hello” Job with a green tick
Select Branch and option circleci-project-setup
Step 4 - Break your build
In this section, you will edit the .circleci/config.yml file and see what happens if a build does not complete successfully.
It is possible to edit files directly on GitHub.

The GitHub file editor should look like this
Scroll down and Commit your changes on GitHub
After committing your changes, then return to the Projects page in CircleCI. You should see a new pipeline running… and it
will fail! What’s going on? The Node orb runs some common Node tasks. Because you are working with an empty
repository, running npm run test, a Node script, causes the configuration to fail. To fix this, you need to set up a Node
project in your repository.
Step 5 – Use Workflows
You do not have to use orbs to use CircleCI. The following example details how to create a custom configuration that also
uses the workflow feature of CircleCI.
1) Take a moment and read the comments in the code block below. Then, to see workflows in action, edit
your .circleci/config.yml file and copy and paste the following text into it.
You don’t need to write the comments which are the text after #
2) Commit these changes to your repository and navigate back to the CircleCI Pipelines page. You should see your pipeline
running.
3) Click on the running pipeline to view the workflow you have created. You should see that two jobs ran (or are currently
running!) concurrently.
Step 5 – Add some changes to use workspaces
Each workflow has an associated workspace which can be used to transfer files to downstream jobs as the workflow
progresses. You can use workspaces to pass along data that is unique to this run and which is needed for downstream
jobs. Try updating config.yml to the following:
Updated config.yml in GitHub file editor should be updated like this
Finally your workflow with the jobs running should look like this
