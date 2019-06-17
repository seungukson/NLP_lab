SELECT 
    *
FROM
    sample54;

SELECT 
    MIN(a)
FROM
    sample54;
SELECT 
    *
FROM
    sample54
WHERE
    a = (SELECT 
            MIN(a)
        FROM
            sample54);

SELECT 
    *
FROM
    sample54
WHERE
    no = (SELECT 
            MIN(no)
        FROM
            sample54);

SELECT 
    *
FROM
    sample54;
SELECT 
    *
FROM
    sample54
WHERE
    a > (SELECT AVG(sample54.a));

SELECT 
    (SELECT 
            COUNT(*)
        FROM
            sample51) AS len_s51,
    (SELECT 
            COUNT(*)
        FROM
            sample54) AS len_s54;
            
SELECT 
    *
FROM
    sample54;
UPDATE sample54 
SET 
    a = (SELECT 
            MAX(a)
        FROM
            sample34);

SELECT 
    *
FROM
    sample54;
SELECT 
    *
FROM
    sample34;
SELECT 
    *
FROM
    (SELECT 
        *
    FROM
        sample34) pr;

SELECT 
    *
FROM
    (SELECT 
        *
    FROM
        sample54
    ORDER BY no DESC) s54
WHERE
    no >= 4;
    
SELECT 
    *
FROM
    sample541;
SELECT 
    *
FROM
    sample54;
SELECT 
    *
FROM
    sample51;

-- sample54의 길이와 sample51의 길이를 sample541의 a,b열에 각각 추가해보세요. 
insert into sample541 values ((select count(*) from sample54), (select count(*) from sample51));

SELECT 
    *
FROM
    sample541;

-- value 대신 select 이용해서 insert사용

insert into sample541 values (1,2); 
insert into sample541 select 1,2;

SELECT 
    *
FROM
    sample551;
    
insert into sample552 select no from sample551;
select * from sample552;

insert into sample54 select no, quantity from sample51;
select * from sample54;

-- 상관서브쿼리
-- 서로다른 테이블의 데이터를 서로 비교하는 서브쿼리  
select * from sample551;
select * from sample552;
# sample522를 조회해서 sample552에 있는 no 값만 sample551에 업데이트하고 싶다면?
update sample551 set a=1 where exists (select * from sample552 where no2=no);

update sample551 set a=0 where not exists (select * from sample552 where no2=no);