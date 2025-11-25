let boids = [];
let numBoids = 30; // fewer birds
let birdImg;

let video;
let poseNet;
let poses = [];

function preload() {
  birdImg = loadImage("bird.gif"); // Make sure this file exists in your project folder
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  noSmooth();

  // Initialize boids
  for (let i = 0; i < numBoids; i++) {
    boids.push(new Boid(random(width), random(height)));
  }

  // Setup webcam
  video = createCapture(VIDEO);
  video.size(width, height);
  video.hide();

  // Initialize PoseNet using the raw video element
  poseNet = ml5.poseNet(video.elt, () => console.log("PoseNet ready"));
  poseNet.on("pose", (results) => {
    poses = results;
  });
}

function draw() {
  background(255);

  // Draw mirrored webcam
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);
  pop();

  // Update & draw boids
  for (let b of boids) {
    b.flock(boids);
    b.bodyRepel(); // repel off body including head and legs
    b.update();
    b.edges();
    b.show();
  }
}

// -----------------
// Boid class
// -----------------
class Boid {
  constructor(x, y) {
    this.pos = createVector(x, y);
    this.vel = p5.Vector.random2D().setMag(random(1, 2));
    this.acc = createVector();
    this.maxForce = 0.2;   // steering limits
    this.maxSpeed = 3;
  }

  flock(boids) {
    let sep = this.separate(boids).mult(1.5);
    let ali = this.align(boids).mult(1.0);
    let coh = this.cohesion(boids).mult(1.0);
    this.acc.add(sep);
    this.acc.add(ali);
    this.acc.add(coh);
  }

  bodyRepel() {
    if (!poses || poses.length === 0) return;

    let pose = poses[0].pose;

    // Keypoints for torso, head, and legs
    let keypoints = [
      pose.leftShoulder,
      pose.rightShoulder,
      pose.leftHip,
      pose.rightHip,
      pose.nose,
      pose.leftKnee,
      pose.rightKnee,
      pose.leftAnkle,
      pose.rightAnkle
    ];

    for (let kp of keypoints) {
      if (!kp) continue;

      // Mirror X because video is mirrored
      let kpX = map(kp.x, 0, video.width, width, 0);
      let kpY = kp.y;
      let bodyPos = createVector(kpX, kpY);

      let d = dist(this.pos.x, this.pos.y, bodyPos.x, bodyPos.y);

      // STRONGER force + bigger radius
      let radius = 180;
      if (d < radius) {
        let force = p5.Vector.sub(this.pos, bodyPos);
        force.setMag((radius - d) * 0.45);  // stronger push
        this.acc.add(force);
      }
    }
  }

  separate(boids) {
    let desiredSeparation = 25;
    let steer = createVector();
    let total = 0;
    for (let other of boids) {
      let d = this.pos.dist(other.pos);
      if (d > 0 && d < desiredSeparation) {
        let diff = p5.Vector.sub(this.pos, other.pos);
        diff.normalize();
        diff.div(d);
        steer.add(diff);
        total++;
      }
    }
    if (total > 0) {
      steer.div(total);
      steer.setMag(this.maxSpeed);
      steer.sub(this.vel);
      steer.limit(this.maxForce);
    }
    return steer;
  }

  align(boids) {
    let perception = 50;
    let avg = createVector();
    let total = 0;
    for (let other of boids) {
      let d = this.pos.dist(other.pos);
      if (d < perception && other !== this) {
        avg.add(other.vel);
        total++;
      }
    }
    if (total > 0) {
      avg.div(total);
      avg.setMag(this.maxSpeed);
      let steer = p5.Vector.sub(avg, this.vel);
      steer.limit(this.maxForce);
      return steer;
    }
    return createVector();
  }

  cohesion(boids) {
    let perception = 50;
    let center = createVector();
    let total = 0;
    for (let other of boids) {
      let d = this.pos.dist(other.pos);
      if (d < perception && other !== this) {
        center.add(other.pos);
        total++;
      }
    }
    if (total > 0) {
      center.div(total);
      return this.seek(center);
    }
    return createVector();
  }

  seek(target) {
    let desired = p5.Vector.sub(target, this.pos);
    desired.setMag(this.maxSpeed);
    let steer = p5.Vector.sub(desired, this.vel);
    steer.limit(this.maxForce);
    return steer;
  }

  update() {
    this.vel.add(this.acc);
    this.vel.limit(this.maxSpeed);
    this.pos.add(this.vel);
    this.acc.mult(0);
  }

  edges() {
    if (this.pos.x > width) this.pos.x = 0;
    if (this.pos.x < 0) this.pos.x = width;
    if (this.pos.y > height) this.pos.y = 0;
    if (this.pos.y < 0) this.pos.y = height;
  }

  show() {
    push();
    translate(this.pos.x, this.pos.y);
    rotate(this.vel.heading() + PI);
    imageMode(CENTER);
    noSmooth();
    image(birdImg, 0, 0, 100, 100); // bigger birds
    pop();
  }
}
