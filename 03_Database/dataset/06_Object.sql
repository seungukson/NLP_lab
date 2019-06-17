select database();
use sample;
CREATE TABLE sample62 (
    no INTEGER NOT NULL,
    a VARCHAR(30),
    b DATE
);
show tables;
desc sample62;

insert into sample62 values (1,'son','2019-06-17');
select * from sample62;

#delete보다 빠르다.
truncate table sample62;
select * from sample62;

drop table sample623;

create table student_table(
no integer(11) not null,
name varchar(10)
);

select * from student_table;

alter table student_table add birth date;
-- birth열의 속성을 varchar로 바꿔보자
desc student_table;
alter table student_table modify birth varchar(10); 
alter table student_table change date date varchar(10) not null;
-- alter table student_table rename column date to birth;
-- change [기존열이름] [바꿀이름] [속성]

-- date 열의 속성에 not null 추가 
alter table student_table modify date varchar(10) not null;
-- date열 속성에 not null을 다시 삭제?
alter table student_table modify date varchar(10);

-- date 열을 다시 birth로 바꾸고, 속성은 date로.alter
alter table student_table change date birth date;

-- 인덱스
create index student_index on student_table(no);
select * from student_table;

insert into student_table values (1,'son','2019-06-12'),(2,'seunguk','2019-06-13');

-- EXPLAIN(possible keys, key) : 인덱스 확인 
explain select * from student_table where no> 1;
-- index_student ->NULL로 바뀜.
drop index student_index on student_table;

-- view
create view student_view as select * from student_table;
create view no_name_view as select no, name from student_table;

select * from no_name_view;
explain select * from no_name_view;

drop view student_view;