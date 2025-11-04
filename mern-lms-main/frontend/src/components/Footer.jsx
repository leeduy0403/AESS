import { Footer } from "flowbite-react";
import { Link } from "react-router-dom";
import LMS from "../assets/LMS.png";

export default function FooterCom() {
  return (
    <Footer container className="bg-[#26597C] rounded-none">
      <div className="w-full max-w-7xl mx-auto">
        <div className="grid w-full justify-between sm:flex md:grid-cols-1">
          <div className="mt-5">
            <Link to={"/"}>
              <img src={LMS} alt="logo" className="w-[80px]" />
            </Link>
            <Footer.LinkGroup col className="mt-1">
              <Footer.Link
                href="https://google.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white"
              >
                <i className="fa-solid fa-location-dot mr-2"></i>
                Cơ sở 1: 268 Lý Thường Kiệt, Phường 14, Quận 10, Thành phố Hồ
                Chí Minh, Việt Nam
              </Footer.Link>
              <Footer.Link
                href="/about"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white"
              >
                <i className="fa-solid fa-location-dot mr-2"></i>
                Cơ sở 2: Khu đô thị Đại học Quốc Gia Tp.HCM, Thủ Đức
              </Footer.Link>
            </Footer.LinkGroup>
          </div>
          <div className="grid grid-cols-2 gap-8 mt-4 sm:grid-cols-3 sm:gap-6">
            <div>
              <Footer.Title
                title="About Us"
                className="text-white normal-case text-lg"
              />
              <Footer.LinkGroup col>
                <Footer.Link
                  href="https://google.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white"
                >
                  Who we are
                </Footer.Link>
                <Footer.Link
                  href="/about"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white"
                >
                  Our Services
                </Footer.Link>
                <Footer.Link
                  href="/about"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white"
                >
                  Guidelines
                </Footer.Link>
              </Footer.LinkGroup>
            </div>
            <div>
              <Footer.Title
                title="Help Center"
                className="text-white normal-case text-lg"
              />
              <Footer.LinkGroup col>
                <Footer.Link
                  href="#"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white"
                >
                  Email: helpcenter@gmail.com
                </Footer.Link>
                <Footer.Link
                  href="/about"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white"
                >
                  Hotline: 0123456789
                </Footer.Link>
              </Footer.LinkGroup>
            </div>
            <div>
              <Footer.Title
                title="Contact Us"
                className="text-white normal-case text-lg"
              />
              <Footer.LinkGroup col>
                <Footer.Link
                  href="#"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white"
                >
                  Email: contact@gmail.com
                </Footer.Link>
                <Footer.Link
                  href="/about"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-white"
                >
                  Phone number: 0987654321
                </Footer.Link>
              </Footer.LinkGroup>
            </div>
          </div>
        </div>
        <Footer.Divider />
        <Footer.Copyright
          href="#"
          by="Learning Management System"
          year={new Date().getFullYear()}
          className="text-white"
        />
      </div>
    </Footer>
  );
}
