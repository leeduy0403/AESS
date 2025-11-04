import { Swiper, SwiperSlide } from "swiper/react";
import SwiperCore from "swiper";
import { Navigation } from "swiper/modules";
import "swiper/css/bundle";
import { Link } from "react-router-dom";
import LMS from "../assets/LMS.png";
import { Button } from "@mui/material";

export default function Home() {
  SwiperCore.use([Navigation]);
  return (
    <div className="min-h-screen">
      <div className="relative inset-0 z-10">
        <Swiper navigation>
          <SwiperSlide>
            <div
              className=""
              style={{
                background: `url("https://hcmut.edu.vn/img/carouselItem/36986508.jpeg?t=36986508") no-repeat`,
                backgroundSize: "cover",
                height: "100vh",
                backgroundPosition: "50% 70%",
              }}
            ></div>
          </SwiperSlide>
          <SwiperSlide>
            <div
              className=""
              style={{
                background: `url("https://congchungnguyenhue.com/Uploaded/Images/Original/2023/11/29/img-0130-398_2911104836.jpg") no-repeat`,
                backgroundSize: "cover",
                height: "100vh",
                backgroundPosition: "50% 85%",
              }}
            ></div>
          </SwiperSlide>
          <SwiperSlide>
            <div
              className=""
              style={{
                background: `url("https://lockernlock.vn/wp-content/uploads/2023/09/LnL-khuon-vien-truong-Bach-Khoa.jpg") center no-repeat`,
                backgroundSize: "cover",
                height: "100vh",
              }}
            ></div>
          </SwiperSlide>
        </Swiper>
        <div
          className="absolute inset-0 flex items-center justify-center z-20"
          style={{
            pointerEvents: "none", // Makes the overlay itself non-interactive
          }}
        >
          <div
            className="flex bg-white p-8 rounded-lg bg-opacity-80 lg:w-4/5 lg:h-3/5 items-center justify-center gap-36"
            style={{
              pointerEvents: "auto", // Enable interactions inside the overlay
            }}
          >
            <div className="">
              <Link to={"/"}>
                <img src={LMS} alt="logo" className="w-80 mx-auto lg:mx-0" />
              </Link>
              <p className="text-2xl font-bold mt-4 text-gray-800">
                Connect, Learn, and Succeed with{" "}
                <span className="text-[#26597C]">SmartLMS</span>
              </p>
            </div>
            <div className="flex flex-col gap-4 items-center text-center lg:text-left">
              <h1 className="text-4xl font-bold text-[#26597C]">
                Welcome to SmartLMS
              </h1>
              <h1 className="text-lg font-medium text-gray-700">
                Software developed by <strong>HCMUT</strong>
              </h1>
              <div className="w-full max-w-sm">
                <Link to={"/classes"}>
                  <Button
                    variant="contained"
                    style={{
                      backgroundColor: "#26597C",
                      textTransform: "none",
                    }}
                    size="large"
                    fullWidth
                  >
                    Get Started
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
