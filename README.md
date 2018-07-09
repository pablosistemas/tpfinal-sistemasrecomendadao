# Recommendation Systems 
### Track reproducibility
[Local Item-Item Models for Top-N Recommendation](http://delivery.acm.org/10.1145/2960000/2959185/p67-christakopoulou.pdf?ip=200.233.214.199&id=2959185&acc=CHORUS&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1531162836_0122336764f77688d215069120dcb54f)

### References
[SLIM](https://www-users.cs.umn.edu/~ningx005/slim/html/) Home
[SLIM](https://www-users.cs.umn.edu/~ningx005/slim/manual.pdf) Manual

## **Installation**
### SLIM and CLUTO
CLUTO: *[Link](http://glaros.dtc.umn.edu/gkhome/fetch/sw/cluto/cluto-2.1.1.tar.gz)*
SLIM: *[Link](https://www-users.cs.umn.edu/~ningx005/slim/slim-1.0.tar.gz)*

#### Installation

> mkdir path/to/desired/installation/folder 
> cd path/to/desired/installation/folder

Move the downloaded SLIM to the created path

> tar -zxf slim-1.0.tar.gz
> cd slim-1.0/build
> cmake ..

##### The next steps are some dependencies I would to solve manually:
The GNU Scientific Library for numerical analysis:
> yum install gsl-devel.x86_64

##### SLIM building and installation:
> make
> make install

##### Inside the build directory you would to see now the examples directory:
> ./examples/slim_learn
> ./examples/slim_predict

`slim_learn` and `slim_predict` are the SLIM binary files.

#### Experiment

$ u[i] $: individual users
$ t[j] $: individual items
$ U $: set of users, |U| = m
$ T $: set of items, |T|=n
$ A $: user-item matrix mxn
$ a[i,j] $: element $(i,j)$ of $A$ is 1 if user $u[i]$ i has ever purchased/rated item $t[j]$, 0 otherwise
$ W $ = $n \times n$ matrix of aggregation coefficients (similarity matrix)
$ Ã $ = $AW$ 

## Dataset links:
- Jester Dataset: [Link](http://www.ieor.berkeley.edu/~goldberg/jester-data/jester-data-1.zip)
- MovieLens10M: [Link](http://files.grouplens.org/datasets/movielens/ml-10m.zip)
- Flixster: [Link](http://socialcomputing.asu.edu/uploads/1296675547/Flixster-dataset.zip)

### Hints (from original paper)
#### Flixter dataset:
**Preprocessing**: Create a subset keeping the users that have rated more than thirty items and the items that have been rated by at least twenty-five users.
#### Netflix dataset:
**Preprocessing**: The subset was created by keeping the users who have rated between thirty and five hundred items
