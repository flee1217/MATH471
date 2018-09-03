++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
README file for Homework 1, Math 471, Fall 2018, Aaron Segura
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- There is a homework report is in the directory.
- There is no code for homework 1 so no hw1/code subdirectory was made
- The purpose of this homework is to familarize the class with Git, working with a shell, and submitting their homework via git.
- This README file contains the instructions to create a README file and report to be pushed to lobogit including the necessary commands
- This assumes that Git has been installed  and a LoboGit repository has been created with the name MATH471Segura

1. The first task is to clone the git repository into the terminal window on your MAC (by Applications>Utilities>Terminal)or Git Bash (if you are using windows)
This can be done by the following command (for example):
$ git clone https://lobogit.unm.edu/enjoi16/MATH471Segura.git

2. Next create a file named test.txt inside the repository directory with a dummy line of text

3. Add, commit, and push this new text file with a commit message like "\My 1st commit of test.txt." to your repository with the following command:
$ git add test.txt
$ git commit -m "My 1st commit of test.txt."
$ git push

4. Now modify your test.txt file by adding two more lines to the test.txt file with the following commands:
I am YOUR_NAME
I am learning version control.

5. Then add and commit the modifeded file, using a commit message like "Added two more lines to
test.txt.". Next, push these changes to your repository.

$ git add test.txt
$ git commit -m "Added two more lines to test.txt."
$ git push

6. Next, add a subdirectory to the first homework called HW1 to the repository and move test.txt to this subdirectory, then add, commit, and push this directory with the folloing commands:
$ git add HW1
$ git mv test.txt HW1/test.txt.
$ git add test.txt
$ git commit -m "moved test.txt to HW1/test.txt."
$ git push

7. Next, create two README.txt files, one in the top-level directory (i.e., math471Segura),
and one in the subdirectory hw1. This README.txt contains information you are reading in the directory and subdirectory HW1. 
The idea is to generate a top-level README.txt file in the master repository (giving general information about the repository 
and its subdirectories) and one low-level README.txt for each homework assignment (describing for instance how to compile and run the
programs for a particular homework). README.txt can be created in Notepad or Notepad++ in windows.

8. Now, add a report subdirectory to HW1/report using the commands from (6.) above. Execute the command
$ git log 
to find the hash tags for the last two commits.

Then, execute the command including the output from the $ git log command above
$ git diff hash_tag_from_last_commit hash_tag_from_two_commits_ago

Include this output in the report.pdf and a description of what this command is doing in the report.
Modify test.txt by adding a line without commiting the new changes, and then type
$ git diff

Include this output in the report.pdf as well describing what this command is doing.

9. Now add, commit and push the report PDF and report subdirectory to the repository.






