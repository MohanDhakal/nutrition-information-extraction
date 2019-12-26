<html>
<body>
<form method="post">  
Enter Number:  
<input type="number" name="number1" /><br><br>  
<input  type="submit" name="submit" value="Check">  
</form>

<?php 

 
function primeCheck($number){ 
    if ($number == 1) 
    return 0; 
    for ($i = 2; $i <= $number/2; $i++){ 
        if ($number % $i == 0) 
            return 0; 
    } 
    return 1; 
} 

if(isset($_POST['submit']))  
{  
    $number =$_POST['number1']; 
    $flag = primeCheck($number); 
if ($flag == 1) 
    echo "$number is prime number."; 
else
    echo "$number is not prime number";
}

?> 
</body>
</html>