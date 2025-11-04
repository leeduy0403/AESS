import { Button, Skeleton, TextField } from "@mui/material";
import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";
import ClassCard from "../components/ClassCard";
import {
  GridViewOutlined as GridViewOutlinedIcon,
  ListAlt as ListAltIcon,
} from "@mui/icons-material";

export default function Classes() {
  const { currentUser } = useSelector((state) => state.user);
  const [searchData, setSearchData] = useState({
    searchTerm: "",
    semester: "",
  });
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isList, setIsList] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const semesters = [242, 241, 233, 232, 231];

  useEffect(() => {
    const urlParams = new URLSearchParams(location.search);
    const searchTermFromUrl = urlParams.get("searchTerm");
    const semesterFromUrl = urlParams.get("semester");
    if (searchTermFromUrl || semesterFromUrl) {
      setSearchData({
        ...searchData,
        searchTerm: searchTermFromUrl,
        semester: semesterFromUrl,
      });
    }
    const fetchClasses = async () => {
      setLoading(true);
      const searchQuery = urlParams.toString();
      let res;
      let data;
      if (!semesterFromUrl) {
        setIsSearching(false);
        if (!searchTermFromUrl && !semesterFromUrl) {
          setIsSearching(false);
          setClasses([]);
        }
        for (const semester of semesters) {
          try {
            const res = await fetch(
              `/api/class/get/${currentUser?._id}?semester=${semester}`
            );
            const data = await res.json();
            if (res.ok && data.length > 0) {
              setClasses((prev) => [...prev, data]);
            }
          } catch (error) {
            console.log(error.message);
          }
        }
      } else {
        setIsSearching(true);
        res = await fetch(`/api/class/get/${currentUser?._id}?${searchQuery}`);
        data = await res.json();
      }
      if (!res?.ok) {
        setLoading(false);
        return;
      }
      if (res.ok) {
        setClasses(data);
        setLoading(false);
        if (data.length === 9) {
          setShowMore(true);
        } else {
          setShowMore(false);
        }
      }
    };
    fetchClasses();
  }, [location.search, currentUser?._id]);

  const handleChange = (e) => {
    if (e.target.id === "searchTerm") {
      setSearchData({ ...searchData, searchTerm: e.target.value });
    }
    if (e.target.id === "semester") {
      setSearchData({ ...searchData, semester: e.target.value });
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const urlParams = new URLSearchParams(location.search);
    urlParams.set("searchTerm", searchData.searchTerm);
    urlParams.set("semester", searchData.semester);
    const searchQuery = urlParams.toString();
    navigate(`/classes?${searchQuery}`);
  };

  const handleShowMore = async () => {
    const numberOfClasses = classes.length;
    const startIndex = numberOfClasses;
    const urlParams = new URLSearchParams(location.search);
    urlParams.set("startIndex", startIndex);
    const searchQuery = urlParams.toString();
    const res = await fetch(`/api/class/get?${searchQuery}`);
    if (!res.ok) {
      return;
    }
    if (res.ok) {
      const data = await res.json();
      setClasses([...classes, ...data]);
      if (data.length === 9) {
        setShowMore(true);
      } else {
        setShowMore(false);
      }
    }
  };

  return loading ? (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="w-full">
          <Skeleton variant="rectangular" width="100%" height={200} />
          <Skeleton width="60%" />
          <Skeleton width="80%" />
        </div>
      ))}
    </div>
  ) : (
    <div className="flex flex-col md:flex-row">
      <div className="w-full">
        <div className="mt-4 mx-auto lg:w-10/12">
          <div className="border-b-2 border-gray-500">
            <h1 className="text-3xl font-semibold my-3 text-[#26597C]">
              My Courses
            </h1>
          </div>
          <form onSubmit={handleSubmit} className="mt-5">
            <div className="flex justify-between items-center gap-2">
              <div className="flex gap-2">
                <TextField
                  type="text"
                  size="small"
                  fullWidth
                  placeholder="Search by code, name"
                  id="searchTerm"
                  autoComplete="off"
                  value={searchData?.searchTerm}
                  onChange={handleChange}
                />
                <TextField
                  size="small"
                  fullWidth
                  placeholder="Semester (e.g. 241)"
                  id="semester"
                  type="text"
                  value={searchData?.semester}
                  onChange={handleChange}
                />
                <div className="flex items-center gap-2">
                  <Button
                    variant="contained"
                    style={{
                      backgroundColor: "#26597C",
                      textTransform: "none",
                    }}
                    type="submit"
                  >
                    Filter
                  </Button>
                  <Button
                    variant="contained"
                    style={{
                      backgroundColor: !isList ? "#26597C" : "#FFFFFF",
                    }}
                    onClick={() => setIsList(false)}
                  >
                    <GridViewOutlinedIcon
                      style={{ color: !isList ? "" : "#000000" }}
                    />
                  </Button>
                  <Button
                    variant="contained"
                    style={{
                      backgroundColor: isList ? "#26597C" : "#FFFFFF",
                    }}
                    onClick={() => setIsList(true)}
                  >
                    <ListAltIcon style={{ color: isList ? "" : "black" }} />
                  </Button>
                </div>
              </div>
            </div>
          </form>
        </div>
        <div className="mt-5 flex flex-wrap mx-auto lg:w-10/12">
          {!loading && classes.length === 0 && (
            <p className="text-xl text-gray-500">No classes found.</p>
          )}
          {loading && <p className="text-xl text-gray-500">Loading...</p>}
          {!loading && classes && semesters && isSearching ? (
            <div className="w-full flex flex-col mx-auto gap-4 border-t-2 border-gray-500">
              <h1 className="text-3xl font-semibold mt-3 text-[#26597C]">
                Semester {classes[0]?.semester} | Academic year 2024-2025
              </h1>
              {!isList ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                  {classes?.length > 0 &&
                    classes.map((classItem, index) => (
                      <ClassCard
                        key={index}
                        classId={classItem?._id}
                        isList={isList}
                      />
                    ))}
                </div>
              ) : (
                <div className="w-full flex flex-col gap-4 mb-4 mx-auto">
                  {classes?.length > 0 &&
                    classes.map((classItem, index) => (
                      <ClassCard
                        key={index}
                        classId={classItem?._id}
                        isList={isList}
                      />
                    ))}
                </div>
              )}
            </div>
          ) : (
            <div className="w-full mx-auto">
              {classes?.length > 0 &&
                classes.map((classesSection, index) => (
                  <div
                    key={index}
                    className="flex flex-col gap-4 border-t-2 border-gray-500"
                  >
                    <h1 className="text-3xl font-semibold mt-3 text-[#26597C]">
                      Semester {semesters[index]} | Academic year 2024-2025
                    </h1>
                    {!isList ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-4">
                        {classesSection?.length > 0 &&
                          classesSection.map((classItem) => (
                            <ClassCard
                              key={classItem?._id}
                              classId={classItem?._id}
                              isList={isList}
                            />
                          ))}
                      </div>
                    ) : (
                      <div className="w-full flex flex-col gap-4 mb-4 mx-auto">
                        {classesSection?.length > 0 &&
                          classesSection.map((classItem) => (
                            <ClassCard
                              key={classItem?._id}
                              classId={classItem?._id}
                              isList={isList}
                            />
                          ))}
                      </div>
                    )}
                  </div>
                ))}
            </div>
          )}
          {showMore && (
            <button
              onClick={handleShowMore}
              className="text-teal-500 text-lg hover:underline p-7 w-full"
            >
              Show More
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
