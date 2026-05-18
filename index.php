<?php

$uri = "postgres://avnadmin:<redacted>@pg-9cda73c-inseec-ce6c.e.aivencloud.com:17868/defaultdb?sslmode=require";

$fields = parse_url($uri);

// build the DSN including SSL settings
$conn = "pgsql:";
$conn .= "host=" . $fields["host"];
$conn .= ";port=" . $fields["port"];;
$conn .= ";dbname=defaultdb";
$conn .= ";sslmode=verify-ca;sslrootcert=ca.pem";

$db = new PDO($conn, $fields["user"], $fields["pass"]);

foreach ($db->query("SELECT VERSION()") as $row) {
	print($row[0]);
}

?>