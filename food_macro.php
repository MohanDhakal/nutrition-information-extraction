<?php
include_once('db_connect.php');

$fname="";
$fid=1;
$fat=1.0;
$carbs=1.0;
$protein=1.0;
$units="";

if(isset($_POST["fName"])){
    $fname=$_POST["fName"];
}
if(isset($_POST["fid"])){
    $fid=$_POST["fid"];

}
if(isset($_POST["fat"])){
    $fat=$_POST["fat"];

}
if(isset($_POST["carbs"])){
    $carbs=$_POST["carbs"];
}
if(isset($_POST["protein"])){
    $protien=$_POST["protein"];

}
if(isset($_POST["units"])){
    $units=$_POST["units"];

}


$merosql="Insert Into foodtable (fName,fid,fat,carbs,protein,units)VALUES('$fname','$fid','$fat','$carbs','$protein','$units')";
mysqli_query($conn,$merosql);

if ($conn->query($merosql) === TRUE) {
    echo "New record created successfully";
} else {
    echo "Error: " . $merosql . "<br>" . $conn->error;
}
?>