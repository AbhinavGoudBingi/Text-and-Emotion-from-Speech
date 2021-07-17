#!/usr/bin/bash


## Deal with github's change form master to main
git push --set-upstream origin master
git branch -m master main
git push -u origin main
git symbolic-ref refs/remotes/origin/HEAD refs/remotes/origin/main
git push origin --delete master
