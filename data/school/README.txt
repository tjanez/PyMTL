Splits for the school data (from http://www.cmm.bristol.ac.uk/)
See [H. Goldstein,Multilevel modelling of survey data. The Statistician, 1991, vol. 40, pp. 235-244]

     The dataset come from the Inner London Education Authority (ILEA),
consisting of examination records from 140 secondary schools in years 1985,
1986 and 1987. It is a random 50% sample with 15362 students. The data have
been used to study the effectiveness of schools. The original file has the following format:

     Column    Description              Coding

     1         Year                     1985=1; 1986=2; 1987=3

     2-4       School                   Codes 1-139

     5-6       Exam Score               Numeric score

     7-8       % FSM                    Percent. students eligible for free
                                        school meals

     9-10      % VR1 band               Percent. students in school in VR band
                                        1

     11        Gender                   Male=0; Female=1

     12        VR band of student       VR1=2; VR2=3; VR3=1

     13-14     Ethnic group of          ESWI=1*; African=2; Arab=3;
               student                  Bangladeshi=4; Caribbean=5;
                                        Greek=6;Indian=7;Pakistani=8;
                                        S.E.Asian=9;Turkish=10; Other=11

                                        Mixed=1; Male=2; Female=3
     15        School gender
                                        Maintained=1; Church of
     16        School denomination      England=2; Roman Catholic=3



      *  ESWI: Students born in England, Scotland, Wales or Ireland.

School is the task. Exam score is the output. The categorical
attributes (everything that is not a percentage) have been expressed
as binary attributes. The resulting data set has 27 attributes + 1
bias attribute (see school_b.mat).

task_indexes = starting index for each task

Splits are in school_$i$_indexes.mat (i=1,...,10)

tr = training set indexes (for x in school_b.mat)
tst = test set indexes (for x in school_b.mat)
tr_indexes = starting index for each task in tr
tst_indexes = starting index for each task in tst

