<?php
include_once('db_connect.php');

$fname="";
$fid=1;
$fat=1.0;
$carbs=1.0;
$protein=1.0;
$units="";

if(isset($_GET['fname'])){
    $fname=$_GET["fname"];
}

if(isset($_GET["fid"])){
    $fid=$_GET["fid"];

}
if(isset($_GET["fat"])){
    $fat=$_GET["fat"];

}
if(isset($_GET["carbs"])){
    $carbs=$_GET["carbs"];
}
if(isset($_GET["protein"])){
    $protien=$_GET["protein"];

}
if(isset($_GET["units"])){
    $units=$_GET["units"];

}


$merosql="Insert Into foodtable (fName,fid,fat,carbs,protein,units)VALUES('$fname','$fid','$fat','$carbs','$protein','$units')";
mysqli_query($conn,$merosql);

// if ($conn->query($merosql) === TRUE) {
//     echo "New record created successfully";
// } else {
//     echo "Error: " . $merosql . "<br>" . $conn->error;
// }
?>