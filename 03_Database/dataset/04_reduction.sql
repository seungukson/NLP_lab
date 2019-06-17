 -- count / sum / avg/ min/ max/ 
-- group by 에 의한 집계 / having 

select database();

select * from sample54;
-- count 
select count(*) from sample54;
-- count로 테이블의 row 개수 계산 가능

-- 특정조건을 충족하는 row개수?
-- a<100 인 개수

select count(*) from sample54 where a<100;
desc sample54;

insert into sample54(no) values (5);
select count(a) from sample54; -- null은 count에서 제외;

select * from sample51;
-- 중복 제거
 select distinct name from sample51;
 
 -- name의 고유값의 개수를 카운트?
 select count(distinct name) from sample51;
-- sum(합계)
select * from sample51;
select sum(quantity) from sample51;
select sum(dltflg) from sample51;
select sum(name) from sample51;
-- avg(평균)
SELECT 
    AVG(quantity), sum(quantity)/count(quantity) from sample51;
-- min, max( 최소, 최대)

select min(quantity), max(quantity) from sample51; 

-- groupby by : 특정 열을 기준으로 집계를 하고자 할 때
SELECT 
    *
FROM
    sample51;
    
SELECT 
    name, AVG(quantity) as cdg
FROM
    sample51
GROUP BY name having cdg > 100;

select * from sample51;

SELECT 
    MIN(no), SUM(quantity)
FROM
    sample51
GROUP BY name;

select name, min(no) as minimum, sum(quantity), avg(dltflg) as isDel from sample51 group by name order by name desc;




