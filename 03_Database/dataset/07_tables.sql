select database();

-- 합집합
select * from sample71_a;
select * from sample71_b; 

select * from sample71_b union select * from sample71_a;

select * from sample31;
-- select * from sample31; union select * from sample71_a; # 형태가 달라서 안된다.

select age from sample31 union select * from  sample71_a order by age;

-- union 정렬

select * from sample71_a union select age from sample31 order by a;
select a as num from sample71_a union select age as num from sample31 order by num;

-- sample71_a = 1,2,3
-- sample71_b = 2,10,11
-- union : 1,2,3,10,11
-- 문약 1,2,3,2,10,11과 같이 중복을 허용하고 싶다면?
select a as num from sample71_a union all select b as num from sample71_b;

-- 교집합 : intersect
-- 차집합 : except(or minus)

-- (select a from sample71_a) intersect (select * from sample71_b);

-- union vs 교차결합
select * from sample72_x union select * from sample72_y;
select * from sample72_y;
select * from sample72_x, sample72_y; #열을 그대로 두고 곱하는식.

-- inner join
create table item(
code char(4) not null, 
name varchar(10), 
price integer, 
category varchar(10)
);

create table stock(
code char(4),
date date
);
insert into item values ('1','son',100,'s'),( '2','bag',200,'b'), ('3','phone',1000,'p');
select * from item;

insert into stock values ('1','2019-05-12'), ('2','2018-01-29'), ('3','2011-12-11');
select * from stock;

SELECT 
    name, date
FROM
    item
        INNER JOIN
    stock ON item.code = stock.code;
    
insert into item values ('4','clock',1000,'c');
select * from item;

insert into stock values ('5', '2019-05-19');
select * from stock;

SELECT 
    item.name, stock.date
FROM
    item
        RIGHT JOIN
    stock ON item.code = stock.code;

