<?php
include_once('db_connect.php');

$fname="";
$fid=1;
$fat=1.0;
$carbs=1.0;
$protein=1.0;
$units="";





//$merosql="Insert Into foodtable (fName,fid,fat,carbs,protein,units)VALUES('$fname','$fid','$fat','$carbs','$protein','$units')";

$merosql = "SELECT fName, fat, carbs, protein, units from foodtable where fid=1;";

echo json_encode(mysqli_fetch_assoc(mysqli_query($conn,$merosql)));




// if ($conn->query($merosql) === TRUE) {
//     echo "New record created successfully";
// } else {
//     echo "Error: " . $merosql . "<br>" . $conn->error;
// }
?>