<?php 
// PHP code to check wether a number is prime or Not 
// function to check the number is Prime or Not 
function primeCheck($number){ 
    if ($number == 1) 
    return 0; 
    for ($i = 2; $i <= $number/2; $i++){ 
        if ($number % $i == 0) 
            return 0; 
    } 
    return 1; 
} 
  
// Driver Code 
$number =5; 
$flag = primeCheck($number); 
if ($flag == 1) 
    echo "$number is prime number."; 
else
    echo "$number is not prime number"
?> 