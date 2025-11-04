import { useEffect, useRef, useState } from "react";
import {
  getDownloadURL,
  getStorage,
  ref,
  uploadBytesResumable,
} from "firebase/storage";
import { app } from "../firebase";
import { CircularProgressbar } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import { Alert, Button, Select, MenuItem, FormControl } from "@mui/material";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { Label, TextInput } from "flowbite-react";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";

export default function DashCreateUser() {
  const [faculties, setFaculties] = useState([]);
  const [imageFile, setImageFile] = useState(null);
  const [imageFileUrl, setImageFileUrl] = useState(null);
  const [imageFileUploadProgress, setImageFileUploadProgress] = useState(null);
  const [imageFileUploadError, setImageFileUploadError] = useState(null);
  const [imageFileUploading, setImageFileUploading] = useState(false);
  const [updateUserSuccess, setUpdateUserSuccess] = useState(null);
  const [updateUserError, setUpdateUserError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({});
  const filePickerRef = useRef();
  console.log(formData);

  useEffect(() => {
    const fetchFaculties = async () => {
      try {
        const res = await fetch(`/api/faculty/get`);
        const data = await res.json();
        if (res.ok) {
          setFaculties(data);
        }
      } catch (error) {
        console.log(error.message);
      }
    };
    fetchFaculties();
  }, []);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImageFileUrl(URL.createObjectURL(file));
    }
  };

  useEffect(() => {
    if (imageFile) {
      uploadImage();
    }
  }, [imageFile]);

  const uploadImage = async () => {
    setImageFileUploading(true);
    setImageFileUploadError(null);
    const storage = getStorage(app);
    const fileName = new Date().getTime() + imageFile.name;
    const storageRef = ref(storage, fileName);
    const uploadTask = uploadBytesResumable(storageRef, imageFile);
    uploadTask.on(
      "state_changed",
      (snapshot) => {
        const progress =
          (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
        setImageFileUploadProgress(progress.toFixed(0));
      },
      () => {
        setImageFileUploadError("Could not upload image!");
        setImageFileUploadProgress(null);
        setImageFile(null);
        setImageFileUrl(null);
        setImageFileUploading(false);
      },
      () => {
        getDownloadURL(uploadTask.snapshot.ref).then((downloadUrl) => {
          setImageFileUrl(downloadUrl);
          setFormData({ ...formData, profilePicture: downloadUrl });
          setImageFileUploading(false);
        });
      }
    );
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setUpdateUserSuccess(null);
    setUpdateUserError(null);
    try {
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      if (!res.ok) {
        setUpdateUserError(data.message);
      } else {
        setUpdateUserSuccess("User created successfully!");
      }
      setLoading(false);
    } catch (error) {
      setUpdateUserError(error.message);
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4 w-full">
      <h1 className="my-8 text-center font-bold text-4xl">Profile</h1>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        ref={filePickerRef}
        hidden
      />
      <div
        className="relative w-36 h-36 self-center cursor-pointer shadow-md overflow-hidden rounded-full mx-auto mb-6"
        onClick={() => filePickerRef.current.click()}
      >
        {imageFileUploadProgress && (
          <CircularProgressbar
            value={imageFileUploadProgress || 0}
            text={`${imageFileUploadProgress}%`}
            strokeWidth={5}
            styles={{
              root: {
                width: "100%",
                height: "100%",
                position: "absolute",
                top: 0,
                left: 0,
              },
              path: {
                stroke: `rgba(62, 152, 199, ${imageFileUploadProgress / 100})`,
              },
            }}
          />
        )}
        <img
          src={
            imageFileUrl ||
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAADhCAMAAAAJbSJIAAAAM1BMVEXk5ueutLeqsbTn6eqpr7PJzc/j5ebf4eLZ3N2wtrnBxsjN0NLGysy6v8HT1tissra8wMNxTKO9AAAFDklEQVR4nO2d3XqDIAxAlfivoO//tEOZWzvbVTEpic252W3PF0gAIcsyRVEURVEURVEURVEURVEURVEURVEURVEURVEURflgAFL/AirAqzXO9R7XNBVcy9TbuMHmxjN6lr92cNVVLKEurVfK/zCORVvW8iUBnC02dj+Wpu0z0Y6QlaN5phcwZqjkOkK5HZyPAjkIjSO4fIdfcOwFKkJlX4zPu7Ha1tIcwR3wWxyFhRG6g4Je0YpSPDJCV8a2Sv2zd1O1x/2WMDZCwljH+clRrHfWCLGK8REMiql//2si5+DKWKcWeAGcFMzzNrXC/0TUwQ2s6+LhlcwjTMlYsUIQzPOCb7YBiyHopyLXIEKPEkI/TgeuiidK/R9FniUDOjRDpvm0RhqjMyyXNjDhCfIMYl1gGjIMIuYsnGEYRMRZOMMunaLVwpWRW008v6fYKDIzxCwVAeNSO90BJW6emelYBRF/kHpYGVaoxTDAaxOFsfP9y8hpJ4xd7gOcij7JNGQ1EYFgkPJa1jQEiYZXRaRINKxSDUW9n+FT82lSKadkiru9/4XPqSLWOekGPoY05TAvLm9orm+YWuwHoBHkZKijNBJGmeb61eL6Ff/6q7bLr7yvv3vKGhpDRjvgjGaPz+gUg6YgcvpyAR2FIZ9U6nEEyZRTovmEU32KichpGn7C17XrfyH9gK/c0CMP05HZIM2uf9sEveizKveBy9/6Qt7o89ne33D525cfcIMW6ab+TMEukQbQbu+xu7X3A9bChmWaCeAkG17bpntwXgWxHaMzGPmUaR5dQZiKqRVeUZ3047fi3nAu28h4CHxCsZAgmEH8Y27jJAhm8c+5RQzRQNVGhVFSfxOYIjp/pP7RxzjevYXVGf4eLt+BJ1vCuLuLkrgABgCGXZ2wik5uty+oBvNirI6mkzhAf4Gsb58Hcm67Jzd+KwD10BYPLL3e0MjvKrgAULnOfveF/O4N2Xb9BZom3gJes3F9X5Zze8/6Yt09b4CrqsEjUv8oFBaR2rl+6CZr2xVrp24o/WitBKuGrrpl1+bFkmK2qXTON4VpbdfLa7o7y/WdLxG7lm2Lqh2clOwTegbvc/vj2U78CwhA87Bn8G5Nk3eOb0Nsr9flz3sG78UUtue4kpv1xvjg3TMay62BMlTlP+vrOMnJsRmt/ze0jsfkPPYdAH57hK+34PeOyc8XIXu5xT2HsUkdZz+adwg8HGFfQ3K5jtDvbUiO4Di9/ywHGrL88pDizZ++oTp+an+SMX/ndymUCwmHMdO7yuOx83pUx/eEMU0AvxWndwgidAqOZ8ypCwdEfvvEo6D9HwpA8wzvmOJEqAg9ySu8g4x0Hb9hSB/BANEKJ+LbPBU0lzbAJs4xt1AoshKkUGQmiH8/jJ0gdhTTLmSegHlPE0oOdXALnqDjKYh3px//fSgSWG8UqfrrIICzYYSJXRr9BSPbpNzw7gBjKjKOYI7ReIGqQRIap5+5MdjyvuDkExvGeXSlONWZAP3/AZBwJohU7QJRGU+cTVH18ELmRPNBmibW6MT/k1b0XhdkRBvyT6SB6EYv/GvhSmRNpGngRULsAlxMCGNXp7w3FfdEbTEEDdLI9TdIKRUzUesa3I461ER8cpNT7gMRhpKmYVS9ELOgCUQsa4SsulciKiLbY+AnHD8cpuhISsnxpamI84sbDq9qYJgf8wiiOBrC7Ml7M7ZECCqKoiiKoiiKoiiKoijv5AvJxlZRyNWWLwAAAABJRU5ErkJggg=="
          }
          alt="User avatar"
          className={`rounded-full w-full h-full object-cover border-8 border-[lightgray] ${
            imageFileUploadProgress &&
            imageFileUploadProgress < 100 &&
            "opacity-60"
          }`}
        />
      </div>
      {imageFileUploadError && (
        <Alert severity="error" className="mb-4 border border-red-600">
          {imageFileUploadError}
        </Alert>
      )}
      <form onSubmit={handleSubmit} className="space-y-6">
        <LocalizationProvider dateAdapter={AdapterDayjs}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <Label value="Email" className="text-lg" />
              <TextInput
                required
                id="email"
                sizing="lg"
                placeholder="Enter email: example@example.com"
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Password" className="text-lg" />
              <TextInput
                required
                id="password"
                type="password"
                sizing="lg"
                placeholder="Enter password"
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Name" className="text-lg" />
              <TextInput
                id="name"
                sizing="lg"
                placeholder="Enter name"
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Date of Birth" className="text-lg" />
              <DatePicker
                onChange={(newValue) =>
                  setFormData({
                    ...formData,
                    dateOfBirth: newValue?.format("YYYY-MM-DD"),
                  })
                }
                slotProps={{ textField: { fullWidth: true, size: "large" } }}
                format="DD/MM/YYYY"
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Gender" className="text-lg" />
              <FormControl fullWidth>
                <Select
                  id="gender"
                  onChange={(e) =>
                    setFormData({ ...formData, gender: e.target.value })
                  }
                  size="large"
                >
                  <MenuItem value="male">Male</MenuItem>
                  <MenuItem value="female">Female</MenuItem>
                  <MenuItem value="other">Other</MenuItem>
                </Select>
              </FormControl>
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Address" className="text-lg" />
              <TextInput
                id="address"
                sizing="lg"
                placeholder="Enter address"
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Phone number" className="text-lg" />
              <TextInput
                id="phoneNumber"
                sizing="lg"
                placeholder="Enter phone number"
                onChange={handleChange}
              />
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Enrolled Year" className="text-lg" />
              <FormControl fullWidth>
                <Select
                  id="enrolledYear"
                  onChange={(e) =>
                    setFormData({ ...formData, enrolledYear: e.target.value })
                  }
                  size="large"
                >
                  {[2020, 2021, 2022, 2023, 2024, 2025].map((year) => (
                    <MenuItem key={year} value={year}>
                      {year}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Faculty" className="text-lg" />
              <FormControl fullWidth>
                <Select
                  onChange={(e) =>
                    setFormData({ ...formData, facultyId: e.target.value })
                  }
                >
                  {faculties?.length > 0 &&
                    faculties.map((faculty) => (
                      <MenuItem key={faculty?._id} value={faculty?._id}>
                        {faculty?.name}
                      </MenuItem>
                    ))}
                </Select>
              </FormControl>
            </div>
            <div className="flex flex-col gap-2">
              <Label value="Role" className="text-lg" />
              <FormControl>
                <RadioGroup
                  row
                  aria-labelledby="role"
                  name="row-radio-buttons-group"
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      isAdmin: e.target.value === "admin",
                      isEducator: e.target.value === "educator",
                      isStudent: e.target.value === "student",
                    }))
                  }
                >
                  <div className="flex-1">
                    <FormControlLabel
                      value="admin"
                      control={<Radio />}
                      label="Admin"
                    />
                  </div>
                  <div className="flex-1">
                    <FormControlLabel
                      value="educator"
                      control={<Radio />}
                      label="Educator"
                    />
                  </div>
                  <div className="flex-1">
                    <FormControlLabel
                      value="student"
                      control={<Radio />}
                      label="Student"
                    />
                  </div>
                </RadioGroup>
              </FormControl>
            </div>
            {formData?.isAdmin && (
              <div className="flex flex-col gap-2">
                <Label value="Admin ID" className="text-lg" />
                <TextInput
                  id="adminId"
                  sizing="lg"
                  placeholder="Enter admin ID"
                  onChange={handleChange}
                />
              </div>
            )}
            {formData?.isEducator && (
              <div className="flex flex-col gap-2">
                <Label value="Educator ID" className="text-lg" />
                <TextInput
                  id="educatorId"
                  sizing="lg"
                  placeholder="Enter educator ID"
                  onChange={handleChange}
                />
              </div>
            )}
            {formData?.isStudent && (
              <div className="flex flex-col gap-2">
                <Label value="StudentID" className="text-lg" />
                <TextInput
                  id="studentId"
                  sizing="lg"
                  placeholder="Enter student ID"
                  onChange={handleChange}
                />
              </div>
            )}
          </div>
        </LocalizationProvider>
        <Button
          variant="contained"
          style={{
            backgroundColor: "#26597C",
            color: "#fff",
            textTransform: "none",
          }}
          type="submit"
          size="large"
          disabled={loading || imageFileUploading}
        >
          {loading ? "Creating..." : "Create User"}
        </Button>
        {updateUserSuccess && (
          <Alert severity="success" className="mt-4 border border-green-600">
            {updateUserSuccess}
          </Alert>
        )}
        {updateUserError && (
          <Alert severity="error" className="mt-4 border border-red-600">
            {updateUserError}
          </Alert>
        )}
      </form>
    </div>
  );
}
