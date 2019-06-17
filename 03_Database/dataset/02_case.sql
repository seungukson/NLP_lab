show databases;
use sample;
show tables;
select * from sample21;
select no, name from sample21 where no = 2;
select * from sample21;

select * from sample21;
select * from sample37;

SELECT 
    a,
    CASE
        WHEN a IS NULL THEN 0
        ELSE a
    END 'check_null'
FROM
    sample37;
    
-- 1->남자, 2->여자
 SELECT 
    *
FROM
    sample37;
SELECT 
*,
    CASE
        WHEN a = 1 THEN '남자'
        WHEN a = 2 THEN '여자'
        else '미지정'
    END 'decode'
FROM
    sample37;
    
-- 위 코드를 단순 case문으로 바꿔보세요-- 
SELECT 
    a,
    CASE a
        WHEN 1 THEN '남자'
        WHEN 2 THEN '여자'
      --   WHEN null THEN '미지정', 못쓴다.
        ELSE '미지정'
    END 'decode'
FROM
    sample37;

SELECT 
    a,
    CASE a
        WHEN 1 THEN '남자'
        WHEN 2 THEN '여자'
      --   WHEN null THEN '미지정', 못쓴다.
        ELSE '미지정'
    END 'decode'
FROM
    sample37;
    
select a from sample37;
select coalesce(a) from sample37;

-- select coalesce(