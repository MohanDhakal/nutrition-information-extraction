<html>
<body>
<form method="post">  
Enter Nbr1:  
<input type="nbr" name="nbr1" /><br><br>
Enter Nbr2:
<input type="nbr" name="nbr2" /><br><br>  
<input  type="submit" name="submit" value="Add">  
</form>
<?php
include_once('connectnumber.php');

$nbr1=1;
$nbr2=1;

if(isset($_POST['submit']))  
    {  
        $nbr1 = $_POST['nbr1'];  
        $nbr2 = $_POST['nbr2']; 
        $sum =  $nbr1+$nbr2;
        echo "The sum of $nbr1 and $nbr2 is: ".$sum;   
  
    }

$merosql="Insert Into nbrtable (nbr1,nbr2)VALUES('$nbr1','$nbr2')";
mysqli_query($conn,$merosql);

?>
</body>
</html>