-- 데이터를 추가하고 싶다면?
SELECT DATABASE();
SELECT 
    *
FROM
    sample41;
insert into sample41 values (1, 'abc','2014-01-25');
SELECT 
    *
FROM
    sample41;

-- data 참조 : desc table_name 
desc sample41;

-- 특정 열에만 값을 추가하고 싶다면?--
insert into sample41(no, a) values (2,'def');
SELECT 
    *
FROM
    sample41;
select * from sample41;
alter table sample41 drop c;
insert into sample41(no) values(4);
-- 한번에 여러 행을 추가하고 싶다면?
insert into sample41(no,a) values(3,'g'),(4,'h');
SELECT 
    *
FROM
    sample41; 

-- NOT NULL 제약
insert into sample41 values(5, NULL, NULL);
SELECT 
    *
FROM
    sample41;

 -- default 값이 null 이 아닌 값으로 설정되어 있따면?
SELECT 
    *
FROM
    sample411;
 insert into sample411(no) values (1);
 SELECT 
    *
FROM
    sample411;
 
 desc sample411;
 
-- 열 추가
SELECT *
FROM
    sample41;
alter table sample41 add c varchar(10) NOT NULL;
select * from sample41;
-- 맨 앞에 넣기
alter table sample41 add f int(10) after a;
select * from sample41;

-- data 딸기
select * from sample41;
delete from sample41;

show tables;

select * from sample37;
delete from sample37 where a is NULL;
-- 열 삭제
alter table sample37 drop a;
select * from sample37;
-- 여러 열 삭제

alter table sample41 drop a, drop b;
select * from sample41;


select * from sample411;
alter table sample411 drop d;

select no as asdfasdf from sample411;


-- 여러 열 추가
select * from sample41;
alter table sample41 add (g int(10),h int(10)); 
-- 데이터 수정하기(update)
SELECT 
    *
FROM
    sample51;
desc sample51;
update sample51 set name = 'A',quantity =20 where name ='Z';

 -- no = no+1, quantity = no(변경 후)
 update sample51 set no = no+1,quantity = no; 
 select database();
 
 update sample51 set quantity=3 where no>=7;
 select * from sample51;
 update sample51 set name='B' where name<>'A';
 
 